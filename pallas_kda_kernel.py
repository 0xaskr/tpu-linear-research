import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from typing import Optional

def chunk_gated_delta_rule_fwd_pallas_kernel(
    k_ptr, v_ptr, w_ptr, g_ptr, gk_ptr, h0_ptr, # Inputs
    h_ptr, v_new_ptr, ht_ptr,                  # Outputs
    *,
    T, H, K, V, BT, BV, NT,
    USE_G, USE_GK, USE_INITIAL_STATE, STORE_FINAL_STATE, SAVE_NEW_VALUE, USE_EXP2
):
    # i_nh: index for (Batch * Heads)
    i_nh = pl.program_id(1)
    i_b = i_nh // H
    i_h = i_nh % H

    # Hidden state in VMEM
    # [K, BV]
    b_h = jnp.zeros((K, BV), dtype=jnp.float32)

    if USE_INITIAL_STATE:
        # h0_ptr: [B, H, K, V] -> [1, H, K, BV]
        b_h += h0_ptr[0, i_h, :, :].astype(jnp.float32)

    for i_t in range(NT):
        # Store current state to history
        # h_ptr: [B, H, NT, K, V] -> [1, H, 1, K, BV]
        h_ptr[0, i_h, i_t, :, :] = b_h.astype(h_ptr.dtype)

        start_t = i_t * BT
        
        # Load w for current chunk: [BT, K]
        # w_ptr: [B, H, T, K] -> [1, H, BT, K]
        b_w = w_ptr[0, i_h, start_t : start_t + BT, :] 
        
        # Prediction: v_pred = w @ h
        # [BT, K] @ [K, BV] -> [BT, BV]
        b_v_pred = jnp.matmul(b_w.astype(jnp.float32), b_h)
        
        # Load u (v_ptr) for current chunk: [BT, BV]
        # v_ptr: [B, H, T, V] -> [1, H, BT, BV]
        b_u = v_ptr[0, i_h, start_t : start_t + BT, :]
        
        # Residual: v_new = u - b_v_pred
        b_v_new = b_u.astype(jnp.float32) - b_v_pred
        
        if SAVE_NEW_VALUE:
            v_new_ptr[0, i_h, start_t : start_t + BT, :] = b_v_new.astype(v_new_ptr.dtype)

        if USE_G:
            # g_ptr: [B, H, T] -> [1, H, BT]
            b_g_chunk = g_ptr[0, i_h, start_t : start_t + BT].astype(jnp.float32)
            b_g_last = b_g_chunk[BT - 1]
            
            if USE_EXP2:
                b_decay_v = jnp.exp2(b_g_last - b_g_chunk)
                b_decay_h = jnp.exp2(b_g_last)
            else:
                b_decay_v = jnp.exp(b_g_last - b_g_chunk)
                b_decay_h = jnp.exp(b_g_last)
            
            mask = (start_t + jnp.arange(BT)) < T
            b_decay_v = jnp.where(mask, b_decay_v, 0.0)
            
            b_v_new = b_v_new * b_decay_v[:, None]
            b_h = b_h * b_decay_h

        if USE_GK:
            # gk_ptr: [B, H, T, K] -> [1, H, BT, K]
            b_gk_chunk = gk_ptr[0, i_h, start_t : start_t + BT, :].astype(jnp.float32)
            b_gk_last = b_gk_chunk[BT - 1]
            
            if USE_EXP2:
                b_h = b_h * jnp.exp2(b_gk_last)[:, None]
            else:
                b_h = b_h * jnp.exp(b_gk_last)[:, None]

        # Update state: h = h + k.T @ v_new
        # k_ptr: [B, H, T, K] -> [1, H, BT, K]
        b_k = k_ptr[0, i_h, start_t : start_t + BT, :]
        b_h = b_h + jnp.matmul(b_k.astype(jnp.float32).T, b_v_new.astype(jnp.float32))

    if STORE_FINAL_STATE:
        # ht_ptr: [B, H, K, V] -> [1, H, K, BV]
        ht_ptr[0, i_h, :, :] = b_h.astype(ht_ptr.dtype)

def chunk_gated_delta_rule_fwd_h_jax(
    k : jax.Array, w: jax.Array, u: jax.Array, g: Optional[jax.Array]=None, gk: Optional[jax.Array]=None, initial_state: Optional[jax.Array]=None,
    output_final_state=False, chunk_size=64, save_new_value=True, use_exp2=False
):
    # Inputs: [B, T, H, K]
    # Transpose to [B, H, T, K] for TPU alignment
    k = jnp.transpose(k, (0, 2, 1, 3))
    w = jnp.transpose(w, (0, 2, 1, 3))
    u = jnp.transpose(u, (0, 2, 1, 3))
    if g is not None:
        g = jnp.transpose(g, (0, 2, 1))
    if gk is not None:
        gk = jnp.transpose(gk, (0, 2, 1, 3))
    
    B, H, T, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    BV = 64 # Match Triton
    NT = (T + BT - 1) // BT

    # Define grid: (V blocks, Batch*Heads)
    grid = ((V + BV - 1) // BV, B * H)

    # Handle None inputs for Pallas
    dummy_g = jnp.zeros((B, H, T), dtype=k.dtype) if g is None else g
    dummy_gk = jnp.zeros((B, H, T, K), dtype=k.dtype) if gk is None else gk
    dummy_h0 = jnp.zeros((B, H, K, V), dtype=jnp.float32) if initial_state is None else initial_state

    # Temporary buffers for Pallas if outputs are not needed
    res_h = jnp.empty((B, H, NT, K, V), dtype=k.dtype)
    res_v_new = jnp.empty((B, H, T, V), dtype=u.dtype)
    res_ht = jnp.empty((B, H, K, V), dtype=jnp.float32)
    
    def kernel_wrapper(k_p, v_p, w_p, g_p, gk_p, h0_p, h_p, v_new_p, ht_p):
        chunk_gated_delta_rule_fwd_pallas_kernel(
            k_p, v_p, w_p, g_p, gk_p, h0_p,
            h_p, v_new_p, ht_p,
            T=T, H=H, K=K, V=V, BT=BT, BV=BV, NT=NT,
            USE_G=(g is not None),
            USE_GK=(gk is not None),
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=save_new_value,
            USE_EXP2=use_exp2
        )

    out_h, out_v_new, out_ht = pl.pallas_call(
        kernel_wrapper,
        out_shape=[
            jax.ShapeDtypeStruct(res_h.shape, res_h.dtype),
            jax.ShapeDtypeStruct(res_v_new.shape, res_v_new.dtype),
            jax.ShapeDtypeStruct(res_ht.shape, res_ht.dtype),
        ],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, H, T, K), lambda i_v, i_nh: (i_nh // H, 0, 0, 0)), # k
            pl.BlockSpec((1, H, T, BV), lambda i_v, i_nh: (i_nh // H, 0, 0, i_v)), # u
            pl.BlockSpec((1, H, T, K), lambda i_v, i_nh: (i_nh // H, 0, 0, 0)), # w
            pl.BlockSpec((1, H, T), lambda i_v, i_nh: (i_nh // H, 0, 0)),       # g
            pl.BlockSpec((1, H, T, K), lambda i_v, i_nh: (i_nh // H, 0, 0, 0)), # gk
            pl.BlockSpec((1, H, K, BV), lambda i_v, i_nh: (i_nh // H, 0, 0, i_v)), # h0
        ],
        out_specs=[
            pl.BlockSpec((1, H, NT, K, BV), lambda i_v, i_nh: (i_nh // H, 0, 0, 0, i_v)), # h
            pl.BlockSpec((1, H, T, BV), lambda i_v, i_nh: (i_nh // H, 0, 0, i_v)), # v_new
            pl.BlockSpec((1, H, K, BV), lambda i_v, i_nh: (i_nh // H, 0, 0, i_v)), # ht
        ],
        interpret=True
    )(k, u, w, dummy_g, dummy_gk, dummy_h0)

    # Transpose back to original layout [B, T, H, ...]
    out_h = jnp.transpose(out_h, (0, 2, 1, 3, 4))
    out_v_new = jnp.transpose(out_v_new, (0, 2, 1, 3))
    
    return out_h, (out_v_new if save_new_value else None), (out_ht if output_final_state else None)
