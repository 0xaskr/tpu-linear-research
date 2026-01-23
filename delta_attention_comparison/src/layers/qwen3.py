from typing import Any, cast

import jax
import jax.nn
from flax import nnx
import jax.numpy as jnp

from src.common_types import Config, DType, Array
from src.initializers import nd_dense_init, NdInitializer
from src.linears import DenseGeneral
from src.normalizations import l2norm, Qwen3NextRMSNormGated

def jax_chunk_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 256,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
) -> tuple[Array, None | Array]:
  initial_dtype = query.dtype
  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
  g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

  if pad_size > 0:
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
    g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

  total_sequence_length = sequence_length + pad_size
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1]).astype(jnp.float32))
  query = query * scale

  v_beta = value * jnp.expand_dims(beta, -1)
  k_beta = key * jnp.expand_dims(beta, -1)

  num_chunks = total_sequence_length // chunk_size
  query_c = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  key_c = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  k_beta_c = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  v_beta_c = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
  g_c = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

  mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  
  # --- FUSED SCAN PREPARATION ---
  
  # Prepare scan inputs (transpose to [NumChunks, Batch, Heads, Chunk, Dim])
  # Inputs: query_c, key_c, k_beta_c, v_beta_c, g_cumsum
  def to_scan(x): return jnp.transpose(x, (2, 0, 1, 3, 4))
  def to_scan_4d(x): return jnp.transpose(x, (2, 0, 1, 3)) # For g_cumsum

  output_final_state = initial_state is not None
  if initial_state is None:
    last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
  else:
    last_recurrent_state = initial_state.astype(jnp.float32)

  xs = (
      to_scan(query_c),       # q_i
      to_scan(key_c),         # k_i
      to_scan(k_beta_c),      # k_beta_i
      to_scan(v_beta_c),      # v_beta_i
      to_scan_4d(g_cumsum)    # g_i
  )
  
  mask_inter = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

  def scan_body(prev_state, x):
    q_i, k_i, k_beta_i, v_beta_i, g_i = x
    prec = jax.lax.Precision.HIGHEST
    
    # --- Local Chunk Computation ---
    # Re-compute all intra-chunk terms locally to save memory
    
    # 1. Decay Mask (g_diff)
    # g_i: (B, H, C)
    g_diff = jnp.expand_dims(g_i, -1) - jnp.expand_dims(g_i, -2) # (B, H, C, C)
    g_diff_tril = jnp.tril(g_diff)
    g_diff_exp = jnp.exp(g_diff_tril).astype(jnp.float32)
    decay_mask_i = g_diff_exp
    
    # 2. Local Attention
    # attn = -k_beta @ k.T * decay
    attn_i = -jnp.matmul(k_beta_i, jnp.swapaxes(k_i, -1, -2), precision=prec) * decay_mask_i
    attn_i = jnp.where(mask, 0.0, attn_i)
    
    # 3. Iterative Correction (fori_loop)
    # This logic is purely local to the chunk (C, C)
    # We vmap it over Batch and Heads
    def correct_attn_local(attn_val):
        def inner_attn_body(i, val):
            indices = jnp.arange(chunk_size)
            col_mask = indices < i
            row = val[i, :] * col_mask
            sub_mask = (indices < i)[:, None] & (indices < i)[None, :]
            sub = val * sub_mask
            term = row[:, None] * sub
            summed = jnp.sum(term, axis=0) # sum over rows? No axis=-2 means sum over cols of term?
            # Original: row_exp (C, 1) * sub (C, C) -> term (C, C). Sum axis=-2 (rows) -> (C).
            # Here inputs are (C, C).
            # Let's match original carefully:
            # row_exp = expand_dims(row, -1) -> (C, 1)
            # term = row_exp * sub -> (C, C)
            # summed = sum(term, axis=-2) -> (C) (summing over rows for each col? No axis -2 is rows)
            
            # Original code:
            # row = attn_val[..., i, :] * col_mask
            # ...
            # row_exp = jnp.expand_dims(row, -1)
            # term = row_exp * sub
            # summed = jnp.sum(term, axis=-2)
            
            row_exp = jnp.expand_dims(row, -1)
            term = row_exp * sub
            summed = jnp.sum(term, axis=0) # (C, C) sum axis 0 -> (C)
            
            update_val = row + summed
            original_row = val[i, :]
            new_row = jnp.where(col_mask, update_val, original_row)
            return val.at[i, :].set(new_row)
            
        return jax.lax.fori_loop(1, chunk_size, inner_attn_body, attn_val)

    # vmap over B, H
    attn_i = jax.vmap(jax.vmap(correct_attn_local))(attn_i)
    
    attn_i = attn_i + jnp.eye(chunk_size, dtype=attn_i.dtype)
    
    # 4. Intra-chunk Outputs
    # value_intra = attn @ v_beta
    v_i = jnp.matmul(attn_i, v_beta_i, precision=prec)
    
    # k_cumdecay = attn @ (k_beta * exp(g))
    k_cumdecay_i = jnp.matmul(attn_i, (k_beta_i * jnp.expand_dims(jnp.exp(g_i), -1)), precision=prec)
    
    # --- Recurrent Update ---
    
    # v_new calculation
    v_prime = jnp.matmul(k_cumdecay_i, last_recurrent_state, precision=prec)
    v_new = v_i - v_prime # (B, H, C, D)
    
    # Inter-chunk Attention
    # attn_inter = q * exp(g) @ state
    g_i_exp = jnp.exp(g_i)
    attn_inter = jnp.matmul(q_i * jnp.expand_dims(g_i_exp, -1), last_recurrent_state, precision=prec)
    
    # Core Output
    # core = attn_inter + (attn_local @ v_new)
    # Need attn_local for output.
    # Original: attn_i = q @ k.T * decay.
    # Note: attn_i calculated above was for correction.
    # The attn_i used for v_new (intra) is q @ k.T?
    # No, original code had `attn` (corrected) used for `value_intra`.
    # And `attn_i` (in scan body) = `q @ k.T * decay`.
    # These are different!
    # The `attn` pre-calculated was `K @ K.T`. (The decay matrix logic).
    
    # Let's re-read original:
    # attn (pre-calc) = -k_beta @ k.T * decay ... then corrected.
    # value_intra = attn @ v_beta
    # k_cumdecay = attn @ (k_beta * exp(g))
    
    # THEN in scan_body:
    # attn_i = q @ k.T * decay
    # core = attn_inter + attn_i @ v_new
    
    # So we need to calculate `attn_q` (q@k.T) separately inside scan.
    
    # attn_q for output:
    attn_q = jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2), precision=prec) * decay_mask_i
    attn_q = jnp.where(mask_inter, 0.0, attn_q)
    
    core_attn_out_i = attn_inter + jnp.matmul(attn_q, v_new, precision=prec)
    
    # State Update
    g_i_last_exp = jnp.exp(g_i[..., -1, None, None])
    new_last_recurrent_state = last_recurrent_state * g_i_last_exp

    g_diff_exp_last = jnp.expand_dims(jnp.exp(jnp.expand_dims(g_i[..., -1], -1) - g_i), -1)
    k_i_g_diff = k_i * g_diff_exp_last

    update_term = jnp.matmul(jnp.swapaxes(k_i_g_diff, -1, -2), v_new, precision=prec)
    new_last_recurrent_state = new_last_recurrent_state + update_term

    return new_last_recurrent_state, core_attn_out_i

  scan_body = jax.checkpoint(scan_body)
  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None

class Qwen3NextGatedDeltaNet(nnx.Module):
  def __init__(self, config: Config, dtype: DType = jnp.float32, *, rngs: nnx.Rngs):
    self.config = config
    self.dtype = dtype
    cfg = self.config

    in_features = cfg.emb_dim
    self.num_v_heads = cfg.gdn_num_value_heads
    self.num_k_heads = cfg.gdn_num_key_heads
    self.head_k_dim = cfg.gdn_key_head_dim
    self.head_v_dim = cfg.gdn_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    conv_dim = self.key_dim * 2 + self.value_dim
    conv_kernel_size = cfg.gdn_conv_kernel_dim
    self.v_heads_per_k_head = self.num_v_heads // self.num_k_heads

    self.in_proj_qkvz = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.key_dim * 2 + self.value_dim * 2),
        dtype=cfg.dtype,
        rngs=rngs,
    )
    self.in_proj_ba = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.num_v_heads * 2),
        dtype=cfg.dtype,
        rngs=rngs,
    )

    self.conv1d = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim,
        padding="CAUSAL",
        use_bias=False,
        dtype=cfg.dtype,
        rngs=rngs,
    )
    def a_log_init(key, shape, dtype=jnp.float32):
      a_vals = jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0)
      return jnp.log(a_vals)

    self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_v_heads,)))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (self.num_v_heads,)))

    self.norm = Qwen3NextRMSNormGated(
        num_features=self.head_v_dim,
        eps=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.out_proj = DenseGeneral(
        in_features_shape=self.value_dim,
        out_features_shape=(in_features,),
        dtype=cfg.dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    cfg = self.config
    batch, seq_len, _ = hidden_states.shape

    qkvz = self.in_proj_qkvz(hidden_states)
    ba = self.in_proj_ba(hidden_states)

    new_shape_qkvz = (
        batch,
        seq_len,
        self.num_k_heads,
        2 * self.head_k_dim + 2 * self.head_v_dim * self.v_heads_per_k_head,
    )
    mixed_qkvz = qkvz.reshape(new_shape_qkvz)

    split_indices_qkvz = [
        self.head_k_dim,
        2 * self.head_k_dim,
        2 * self.head_k_dim + (self.v_heads_per_k_head * self.head_v_dim),
    ]
    query, key, value_raw, z_raw = jnp.split(mixed_qkvz, split_indices_qkvz, axis=3)

    value = value_raw.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
    z = z_raw.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

    new_shape_ba = (
        batch,
        seq_len,
        self.num_k_heads,
        2 * self.v_heads_per_k_head,
    )
    mixed_ba = ba.reshape(new_shape_ba)

    split_indices_ba = [self.v_heads_per_k_head]
    b_raw, a_raw = jnp.split(mixed_ba, split_indices_ba, axis=3)

    b = b_raw.reshape(batch, seq_len, self.num_v_heads)
    a = a_raw.reshape(batch, seq_len, self.num_v_heads)

    q = query.reshape(batch, seq_len, -1)
    k = key.reshape(batch, seq_len, -1)
    v = value.reshape(batch, seq_len, -1)

    qkv = jnp.concatenate([q, k, v], axis=-1)

    conv_out = self.conv1d(qkv)
    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)
    q_conv, k_conv, v_conv = jnp.split(qkv_conv, [self.key_dim, 2 * self.key_dim], axis=-1)

    batch, seq_len, _ = hidden_states.shape
    query = q_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
    key = k_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
    value = v_conv.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

    A_log = self.A_log.value
    dt_bias = self.dt_bias.value
    beta = jax.nn.sigmoid(b)
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))
    g = g.astype(cfg.dtype)

    if self.num_v_heads > self.num_k_heads and self.num_v_heads % self.num_k_heads == 0:
      repeats = self.num_v_heads // self.num_k_heads
      query = jnp.repeat(query, repeats, axis=2)
      key = jnp.repeat(key, repeats, axis=2)
    elif self.num_k_heads > self.num_v_heads and self.num_k_heads % self.num_v_heads == 0:
      pass

    core_attn_out, _ = jax_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=cfg.gdn_chunk_size, use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn
    )

    gated_output_reshaped = self.norm(core_attn_out, z)
    gated_output = gated_output_reshaped.reshape(batch, seq_len, -1)
    output = self.out_proj(gated_output)

    return output

def analyze_qwen_operators(config: Config, batch_size: int, seq_len: int):
  try:
    from jax.experimental import roofline
  except ImportError:
    print("jax.experimental.roofline not available.")
    return []

  rngs = nnx.Rngs(0)
  model = Qwen3NextGatedDeltaNet(config=config, rngs=rngs)

  stats = []
  
  # Shapes
  hidden_shape = jax.ShapeDtypeStruct((batch_size, seq_len, config.emb_dim), jnp.float32)
  
  # 1. QKVZ Proj
  _, res_qkvz = roofline.roofline(lambda x: model.in_proj_qkvz(x))(hidden_shape)
  stats.append({"name": "Qwen: QKVZ Proj", "flops": res_qkvz.flops, "bytes": res_qkvz.hbm_bytes})

  # 2. BA Proj
  _, res_ba = roofline.roofline(lambda x: model.in_proj_ba(x))(hidden_shape)
  stats.append({"name": "Qwen: BA Proj", "flops": res_ba.flops, "bytes": res_ba.hbm_bytes})

  # 3. Conv1D
  # Input dim: key_dim*2 + value_dim.
  # k_dim = head_k_dim * num_k_heads
  # v_dim = head_v_dim * num_v_heads
  # But we need to verify actual input size to conv inside __call__.
  # qkv is concat of q, k, v.
  # q: (B, L, K_HEADS * K_DIM)
  # k: same
  # v: (B, L, V_HEADS * V_DIM)
  # So total channels = 2 * (K*K_D) + (V*V_D).
  
  conv_dim = model.key_dim * 2 + model.value_dim
  conv_in_shape = jax.ShapeDtypeStruct((batch_size, seq_len, conv_dim), jnp.float32)
  
  _, res_conv = roofline.roofline(lambda x: model.conv1d(x))(conv_in_shape)
  stats.append({"name": "Qwen: Conv1D", "flops": res_conv.flops, "bytes": res_conv.hbm_bytes})

  # 4. Attn Core
  # Need to construct inputs matching internal shapes.
  # query: (B, L, K_HEADS, K_DIM)
  # key: same
  # value: (B, L, V_HEADS, V_DIM)
  # g: (B, L, V_HEADS)
  # beta: (B, L, V_HEADS)
  
  k_heads = config.gdn_num_key_heads
  v_heads = config.gdn_num_value_heads
  k_dim = config.gdn_key_head_dim
  v_dim = config.gdn_value_head_dim
  
  q_shape = jax.ShapeDtypeStruct((batch_size, seq_len, k_heads, k_dim), jnp.float32)
  v_shape = jax.ShapeDtypeStruct((batch_size, seq_len, v_heads, v_dim), jnp.float32)
  g_shape = jax.ShapeDtypeStruct((batch_size, seq_len, v_heads), jnp.float32)

  # Note: logic handles repeat inside __call__ if v_heads > k_heads, 
  # but jax_chunk_gated_delta_rule expects matched or broadcastable?
  # "if self.num_v_heads > self.num_k_heads ... query = jnp.repeat..."
  # So jax_chunk_gated_delta_rule receives query with V_HEADS if repeated.
  
  # If we pass raw shapes to core, we should mimic what __call__ does.
  # Let's assume we pass the *already repeated* shapes if needed, or rely on core to handle it?
  # No, core doesn't repeat. Core expects compatible shapes.
  # Q and K in core are (B, L, NH, K_DIM).
  # If K_HEADS != V_HEADS, they must be aligned before calling core?
  # __call__ repeats Q and K.
  
  core_q_heads = v_heads if v_heads > k_heads else k_heads
  # If k_heads > v_heads (GQA), then V is repeated? No.
  # The code says:
  # if v > k and v % k == 0: repeat Q, K. (MQA/GQA inverted?)
  # typically GQA means K,V are smaller. Here it seems V is larger?
  # "gdn_num_value_heads"
  
  # Anyway, let's just use core_q_heads.
  
  q_core_shape = jax.ShapeDtypeStruct((batch_size, seq_len, core_q_heads, k_dim), jnp.float32)
  v_core_shape = jax.ShapeDtypeStruct((batch_size, seq_len, v_heads, v_dim), jnp.float32)
  g_core_shape = jax.ShapeDtypeStruct((batch_size, seq_len, v_heads), jnp.float32)
  
  def run_core(q, k, v, g, beta):
      return jax_chunk_gated_delta_rule(
          q, k, v, g, beta, 
          chunk_size=config.gdn_chunk_size, 
          use_qk_norm_in_gdn=config.use_qk_norm_in_gdn
      )

  _, res_core = roofline.roofline(run_core)(q_core_shape, q_core_shape, v_core_shape, g_core_shape, g_core_shape)
  stats.append({"name": "Qwen: Attn Core", "flops": res_core.flops, "bytes": res_core.hbm_bytes})

  # 5. Out Proj
  # Input: (B, L, V_HEADS * V_DIM)
  out_in_shape = jax.ShapeDtypeStruct((batch_size, seq_len, v_heads * v_dim), jnp.float32)
  _, res_out = roofline.roofline(lambda x: model.out_proj(x))(out_in_shape)
  stats.append({"name": "Qwen: Out Proj", "flops": res_out.flops, "bytes": res_out.hbm_bytes})

  return stats