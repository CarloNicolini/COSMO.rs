"""Cross-check COSMO.rs textbook problems against COSMO.jl.

Requires a Julia project with COSMO installed, e.g.:

    julia --project=/tmp/cosmo-compare -e 'using Pkg; Pkg.add(\"COSMO\")'
    julia --project=/tmp/cosmo-compare examples/julia/compare_textbook.jl

Prints status / x / objective for each COSMO.jl example that the Rust
suite ports in `tests/stress_vs_clarabel.rs`.
"""

using COSMO
using SparseArrays
using LinearAlgebra

function report(name, res; xshow=nothing)
    xs = xshow === nothing ? res.x : res.x[xshow]
    println("[$name] status=$(res.status)  obj=$(res.obj_val)  x=$xs")
end

println("=== COSMO.jl textbook examples (reference for COSMO.rs) ===\n")

# --- QP (examples/qp.jl) ---
q = [1; 1.0]
P = sparse([4.0 1; 1 2])
A = [1.0 1; 1 0; 0 1]
l = [1.0; 0; 0]
u = [1; 0.7; 0.7]
Aa = [-A; A]
ba = [u; -l]
model = COSMO.Model()
assemble!(model, P, q, COSMO.Constraint(Aa, ba, COSMO.Nonnegatives);
          settings = COSMO.Settings(verbose = false))
report("qp", COSMO.optimize!(model); xshow=1:2)

# --- LP (examples/lp.jl) ---
c = [1; 2; 3; 4.0]
A = Matrix(1.0I, 4, 4)
b = [10.0; 10; 10; 10]
n = 4
c1 = COSMO.Constraint(-A, b, COSMO.Nonnegatives)
c2 = COSMO.Constraint(Matrix(1.0I, n, n), -ones(n), COSMO.Nonnegatives)
c3 = COSMO.Constraint(1, -5, COSMO.Nonnegatives, n, 2:2)
c4 = COSMO.Constraint([1 0 1 0], -4, COSMO.Nonnegatives)
model = COSMO.Model()
assemble!(model, spzeros(4, 4), c, [c1; c2; c3; c4];
          settings = COSMO.Settings(verbose = false, eps_abs = 1e-4, eps_rel = 1e-5))
report("lp", COSMO.optimize!(model); xshow=1:4)

# --- Box QP (test/UnitTests/qp-box.jl) ---
P = sparse(1.0I, 2, 2)
q = [1.0; -1]
model = COSMO.Model()
assemble!(model, P, q, COSMO.Constraint(sparse(1.0I, 2, 2), zeros(2), COSMO.Box([0.0; 0], [1.0; 1]));
          settings = COSMO.Settings(verbose = false))
report("box_qp", COSMO.optimize!(model); xshow=1:2)

# --- Exp cone (test/UnitTests/exp_cone.jl) ---
P = spzeros(3, 3)
q = [-1.0; 0; 0]
cs1 = COSMO.Constraint(spdiagm(0 => ones(3)), zeros(3), COSMO.ExponentialCone)
cs2 = COSMO.Constraint(SparseMatrixCSC([0 1.0 0; 0 0 1]), [-1.0; -exp(5)], COSMO.ZeroSet)
model = COSMO.Model()
assemble!(model, P, q, [cs1; cs2];
          settings = COSMO.Settings(verbose = false, eps_abs = 1e-4, eps_rel = 1e-4))
report("exp", COSMO.optimize!(model))

# --- Power cone (test/UnitTests/pow_cone.jl) ---
n = 6
P = spzeros(n, n)
q = zeros(6)
q[3] = q[6] = -1
cs1 = COSMO.Constraint(spdiagm(0 => ones(3)), zeros(3), COSMO.PowerCone(0.6), 6, 1:3)
cs2 = COSMO.Constraint(spdiagm(0 => ones(3)), zeros(3), COSMO.PowerCone(0.1), 6, 4:6)
cs3 = COSMO.Constraint([1.0 2.0 0 3.0 0 0], [-3.0], COSMO.ZeroSet)
cs4 = COSMO.Constraint([0 0 0 0 1.0 0], [-1.0], COSMO.ZeroSet)
model = COSMO.Model()
assemble!(model, P, q, [cs1; cs2; cs3; cs4];
          settings = COSMO.Settings(verbose = false, max_iter = 5000))
report("power", COSMO.optimize!(model))

# --- Clarabel-style QP (examples/rust/example_qp.rs) ---
# Clarabel: A_c x + s = b_c, s ∈ K
# COSMO Constraint(A,b): A x + b ∈ K  ⇒  A = −A_c (ineq), b = b_c; for Zero: A = A_c, b = −b_c
P = sparse([6.0 0; 0 4])
q = [-1.0; -4]
c_zero = COSMO.Constraint([1.0 -2.0], [0.0], COSMO.ZeroSet)  # [1,-2]x = 0
c_nn = COSMO.Constraint(
    sparse([-1.0 0; 0 -1; 1 0; 0 1]),   # −A_c for NN rows
    [1.0; 1; 1; 1],
    COSMO.Nonnegatives,
)
model = COSMO.Model()
assemble!(model, P, q, [c_zero; c_nn]; settings = COSMO.Settings(verbose = false))
report("clarabel_qp", COSMO.optimize!(model); xshow=1:2)

println("\nDone.")
