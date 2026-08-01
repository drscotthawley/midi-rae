"""Solver-agnostic guided sampler for the pixel-CFM flow model.

Replaces pcfm_infer's black-box `NeuralODE.trajectory()` with an explicit
integration loop that exposes two hooks -- the single mechanism the paper's
generative section is built on (CFG, inpainting, and cross-block continuation
are all instances of it):

  field(t, x)   -> v    the (possibly guided) velocity field. A `CFGWrapper`
                        is one such field; wrap it further to add SMOOTH
                        guidance (classifier/gradient/plugin). Because the
                        solver evaluates `field` at every substage, smooth
                        guidance stays consistent under higher-order solvers.

  project(t, x) -> x    a HARD-constraint projection applied once after each
                        full step (predict-then-project). Used for inpainting /
                        seam-pinning, where a known region must be held to its
                        value on the flow path. Discontinuous, hence ill-defined
                        inside RK4 substages -> applied per step, not per stage.
                        Left None for pure generation.

Solver: `euler` is the default (fast, and OT-CFM straightens the *unguided*
trajectories so low order suffices). `rk4` is available as an accuracy
cross-check -- expected to agree with euler on the unguided field and to
separate once strong CFG curves it.
"""
import torch

from pcfm_infer import CFGWrapper, IMAGE_SIZE, N_LEVELS


# ---------------------------------------------------------------- integrators
def _euler_step(field, t, x, dt):
    return x + dt * field(t, x)


def _rk4_step(field, t, x, dt):
    k1 = field(t, x)
    k2 = field(t + dt / 2, x + dt / 2 * k1)
    k3 = field(t + dt / 2, x + dt / 2 * k2)
    k4 = field(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


_STEP = {"euler": _euler_step, "rk4": _rk4_step}


def integrate(field, x0, n_steps, solver="euler", project=None, t0=0.0, t1=1.0):
    """Integrate dx/dt = field(t, x) from t0->t1 in n_steps equal steps.

    field:   callable(t: float, x: Tensor) -> Tensor (velocity), same shape as x.
    project: optional callable(t: float, x: Tensor) -> Tensor applied after each
             step AND once at t0 (so the known region is pinned from the start).
    """
    step = _STEP[solver]
    dt = (t1 - t0) / n_steps
    x = x0
    if project is not None:
        x = project(t0, x)
    for k in range(n_steps):
        t = t0 + k * dt
        x = step(field, t, x, dt)
        if project is not None:
            x = project(min(t + dt, t1), x)
    return x


# ------------------------------------------------------------------- fields
def make_cfm_field(model, mlcond, cfg_strength):
    """Wrap the pixel-CFM UNet as field(t, x) with classifier-free guidance.

    Reuses pcfm_infer.CFGWrapper unchanged; only reshapes the scalar solver time
    into the (B,) tensor the UNet's timestep embedding expects."""
    wrapped = CFGWrapper(model, mlcond, float(cfg_strength))

    def field(t, x):
        tt = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(1).expand(x.shape[0])
        return wrapped(tt, x)

    return field


def add_smooth_guidance(field, guide_fn, scale=1.0):
    """Compose an extra SMOOTH guidance term onto an existing field.

    guide_fn(t, x, v) -> Tensor (a velocity increment). This is the socket for
    plugin / classifier / gradient guidance methods -- they return a velocity
    nudge that is added at every solver substage. Returns a new field(t, x)."""
    def guided(t, x):
        v = field(t, x)
        return v + scale * guide_fn(t, x, v)

    return guided


# ---------------------------------------------------- soft (smooth) guidance
def _like(ref, t):
    """Put a captured constraint tensor on the sampler's device/dtype.

    The hooks below are built by the caller (app.py, tests) before the sampler
    has moved anything to the GPU, so x_known/mask are typically CPU float32
    while z_t lives on cuda/mps. Returns t unchanged when already aligned, so
    callers cache the result by rebinding and the per-step cost is one compare."""
    return t if (t.device == ref.device and t.dtype == ref.dtype) else t.to(ref.device, ref.dtype)


def endpoint_estimate(t, x, v):
    """Linear projection to the flow endpoint: x1_hat = x_t + (1-t) v_t."""
    return x + (1.0 - t) * v


def make_soft_inpaint_guidance(x_known, mask, eta=1.0, t_min=0.2, t_max=0.999):
    """SMOOTH inpainting guidance (Pokle-STYLE, but our own time schedule).

    Because we flow in PIXEL space with NO decoder, x1_hat *is* the image, so
    d x1_hat / d z_t = I exactly -> the guidance is closed-form, no autograd:

        x1_hat = x + (1-t) v
        dv     = -eta * (1-t)/t * M^2 * (x1_hat - y)      # y=x_known, M=1 on KNOWN

    mask: (1,1,H,W) 1.0 where KNOWN (matched to y), 0.0 where GENERATED.

    On the (1-t)/t weight: this is NOT Pokle et al.'s schedule -- theirs is
    (1-t)^2/((1-t)^2+t^2) (and a constant 4), both of which failed to work in
    guidance_for_flows.py. The generic classifier-guidance 1/(1-t) also failed
    there. (1-t)/t is the empirically working choice: strong early, ->0 at the
    data end. It diverges as t->0, so guidance is gated off outside
    [t_min, t_max] and paired with a t0~0.2 blended init (guided_generate
    init_t0). This keeps the boundary coherent (no hard seam), unlike
    make_inpaint_project.
    """
    m2 = mask * mask  # mask is 0/1 so this == mask; explicit to mirror the M^2 formula

    def guide(t, x, v):
        nonlocal m2, x_known
        if t < t_min or t > t_max: return torch.zeros_like(v)
        m2, x_known = _like(x, m2), _like(x, x_known)
        x1_hat = endpoint_estimate(t, x, v)
        return -eta * ((1.0 - t) / t) * m2 * (x1_hat - x_known)

    return guide


# ------------------------------------------------------- hard-constraint hooks
def make_inpaint_project(x_known, mask, x0_noise):
    """Predict-then-project hook that pins the KNOWN region to the flow path.

    x_known:  (1,1,H,W) target image in the model's [-1,1] space.
    mask:      (1,1,H,W) 1.0 where KNOWN (held), 0.0 where GENERATED.
    x0_noise:  (1,1,H,W) the fixed t=0 noise for the known region.

    On the linear FM path x_t = (1-t)*x0 + t*x1, the known region at time t is
    (1-t)*noise + t*x_known; we overwrite the masked pixels with it each step."""
    def project(t, x):
        nonlocal x0_noise, x_known, mask
        x0_noise, x_known, mask = _like(x, x0_noise), _like(x, x_known), _like(x, mask)
        known_t = (1.0 - t) * x0_noise + t * x_known
        return mask * known_t + (1.0 - mask) * x

    return project


def make_inpaint_grad(x_known, mask, noise_amp=0.0):
    """Constraint gradient dF/dz1 for PnP-Flow, F = M^2 (z1 - y)^2 (up to a 2).

    Pixel-space + no decoder => the constraint is directly on the endpoint image,
    so this is just the masked residual -- no autograd.

    noise_amp > 0 adds a fresh Gaussian kick inside the hole on every step, so
    structure has something to nucleate around instead of relaxing toward whatever
    the conditioning says. It is confined to the hole by (1 - m2): in the known
    region it would only fight the residual term pulling those pixels back to
    x_known. The caller scales the whole gradient by strength * (1-t)^alpha, so
    the kick anneals to zero as t -> 1 without extra bookkeeping."""
    m2 = mask * mask

    def grad(z1_hat):
        nonlocal m2, x_known
        m2, x_known = _like(z1_hat, m2), _like(z1_hat, x_known)
        g = m2 * (z1_hat - x_known)
        if noise_amp:
            g = g + noise_amp * torch.randn_like(z1_hat) * (1.0 - m2)
        return g

    return grad


# ------------------------------------------------------------ shared setup
def _prepare(demo, img_tensor, drop_levels, seed, device, mlcond=None):
    """Encode conditioning (with optional level drops) and draw the t=0 noise.

    Pass `mlcond` to bypass encoding -- e.g. XMEP-filled conditioning, where the
    hole's embeddings were predicted rather than read off a blanked image."""
    assert demo._loaded, "call demo.load() first"
    if device is not None:
        demo.set_device(device)
    dev = demo.device
    if mlcond is None:
        mlcond = demo.encode_to_mlcond(img_tensor)
    for i in set(int(x) for x in drop_levels):
        mlcond[i] = torch.zeros_like(mlcond[i])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x0 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, generator=gen).to(dev)
    return mlcond, x0, dev


def _to_roll(xT, return_tensor):
    if return_tensor:
        return xT
    return (xT.clip(-1, 1) / 2 + 0.5)[0, 0].detach().cpu().numpy().astype("float32")


# ------------------------------------------------------------ high-level entry
@torch.no_grad()
def guided_generate(demo, img_tensor, drop_levels=(), n_steps=20, seed=0,
                    cfg_strength=4.0, solver="euler", device=None,
                    guide_fn=None, guide_scale=1.0, project=None,
                    init_known=None, init_t0=0.0, return_tensor=False, mlcond=None):
    """pcfm_infer.generate, re-expressed through the explicit integrator.

    With guide_fn=None and project=None this is a drop-in for demo.generate
    (used for the parity check). guide_fn adds SMOOTH guidance (e.g. soft
    inpainting, classifier guidance); project pins a known region via hard
    replacement. Returns a (H,W) [0,1] numpy roll, or the raw (1,1,H,W) [-1,1]
    tensor if return_tensor=True."""
    mlcond, x0, dev = _prepare(demo, img_tensor, drop_levels, seed, device, mlcond)

    # Pokle-style warm start: begin at t0 from a noise<->known blend on the flow
    # path (avoids the (1-t)/t guidance blow-up near t=0).
    if init_known is not None and init_t0 > 0.0:
        x0 = (1.0 - init_t0) * x0 + init_t0 * init_known.to(dev)

    field = make_cfm_field(demo.model, mlcond, cfg_strength)
    if guide_fn is not None:
        field = add_smooth_guidance(field, guide_fn, guide_scale)

    xT = integrate(field, x0, int(n_steps), solver=solver, project=project, t0=float(init_t0))
    return _to_roll(xT, return_tensor)


@torch.no_grad()
def pnpflow_generate(demo, img_tensor, grad_fn, drop_levels=(), n_steps=20, seed=0,
                     cfg_strength=4.0, alpha=0.5, strength=1.0, num_avg=1,
                     device=None, return_tensor=False, mlcond=None):
    """PnP-Flow guidance (Martin et al.): the endpoint estimate z1_hat is the
    primary variable, corrected by the constraint each step then re-projected
    through the flow. Reuses the CFG field. In pixel space grad_fn needs no
    autograd (constraint is directly on the endpoint image).

    Per step (t from 0..1):
        z1*  = z1_hat - strength * (1-t)^alpha * grad_fn(z1_hat)
        z_t  = (1-t) z0 + t z1*
        z1_hat = z_t + (1-t) field(t, z_t)        # optionally averaged over z0 draws
    """
    mlcond, x0, dev = _prepare(demo, img_tensor, drop_levels, seed, device, mlcond)
    field = make_cfm_field(demo.model, mlcond, cfg_strength)

    # extra noise draws for the stability-averaging variant
    noises = [x0]
    for j in range(1, int(num_avg)):
        g = torch.Generator(device="cpu").manual_seed(int(seed) + 1000 + j)
        noises.append(torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g).to(dev))

    z1_hat = x0.clone()
    for k in range(int(n_steps)):
        t = k / int(n_steps)
        z1_star = z1_hat - strength * (1.0 - t) ** alpha * grad_fn(z1_hat)
        acc = 0.0
        for z0 in noises:
            z_t = (1.0 - t) * z0 + t * z1_star
            acc = acc + (z_t + (1.0 - t) * field(t, z_t))
        z1_hat = acc / len(noises)
    return _to_roll(z1_hat, return_tensor)
