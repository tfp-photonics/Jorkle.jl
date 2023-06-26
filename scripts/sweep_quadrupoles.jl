using ArgParse
using HDF5
using Plots

using Jorkle
import Jorkle: c_0

function cmdline_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--npoints"
        arg_type = Int64
        default = 100
        "--beta"
        arg_type = Float64
        default = 0.2
        "--theta"
        arg_type = Float64
        default = pi / 4
        "--ed"
        arg_type = Float64
        default = pi / 6
        "--md"
        arg_type = Float64
        default = pi / 4
    end
    return parse_args(s)
end

function main()
    args = cmdline_args()

    beta = args["beta"]
    theta = args["theta"]
    ed = args["ed"]
    md = args["md"]
    n = args["npoints"]

    nu_max = 2
    h_inc = 1
    phi = 0.0

    gamma = 1 / sqrt(1 - beta^2)
    wi = 1.0

    w0_ratio = 10.0
    w0 = w0_ratio * 2pi * c_0 / wi

    mu, nu = get_munu(nu_max)

    atol = eps(Float64)
    rtol = sqrt(eps(Float64))

    m = Array{Float64}(undef, (n, n))
    rng = collect(LinRange(-π / 2, π / 2, n))
    indices = Iterators.product(eachindex(rng), eachindex(rng))

    for (i, j) in collect(indices)
        me = [ed, rng[i]]
        mm = [md, rng[j]]
        m[i, j] = directivity(
            me, mm, theta, phi, h_inc, wi, w0, gamma, beta, mu, nu, atol, rtol
        )
    end

    h5write("./sweep_results.h5", "m", m)
    h5writeattr("./sweep_results.h5", "m", args)

    heatmap(
        m;
        xlims=(0.5, n + 0.5),
        ylims=(0.5, n + 0.5),
        xticks=([1, n / 2 + 0.5, n], ["-π/2", "0", "π/2"]),
        yticks=([1, n / 2 + 0.5, n], ["-π/2", "0", "π/2"]),
        xlabel="EQ",
        ylabel="MQ",
        colorbar_title="Directivity",
        aspect_ratio=1,
        interpolate=false,
    )
    savefig("sweep_results.png")

    return nothing
end

main()