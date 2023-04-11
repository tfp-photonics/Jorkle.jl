using ArgParse
using JuMP
using Ipopt
using NLopt

using Jorkle
import Jorkle: c_0

function cmdline_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nu"
        arg_type = Int
        default = 2
        "--beta"
        arg_type = Float64
        default = 0.2
        "--theta"
        arg_type = Float64
        default = pi / 4
        "--method"
        arg_type = String
        default = "local"
        range_tester = (x -> x in ["local", "global"])
    end
    return parse_args(s)
end

function main()
    args = cmdline_args()

    nu_max = args["nu"]
    beta = args["beta"]
    theta = args["theta"]
    method = args["method"]

    h_inc = 1
    phi = 0.0

    gamma = 1 / sqrt(1 - beta^2)
    wi = 1.0

    w0_ratio = 10.0
    w0 = w0_ratio * 2pi * c_0 / wi

    mu, nu = get_munu(nu_max)

    atol = eps(Float64)
    rtol = sqrt(eps(Float64))

    function objective(x...)
        s = length(x) ÷ 2
        me = vcat(x[1:s]...)
        mm = vcat(x[(s + 1):end]...)
        d = directivity(me, mm, theta, phi, h_inc, wi, w0, gamma, beta, mu, nu, atol, rtol)
        return d
    end

    model = Model()
    register(model, :objective, 2nu_max, objective; autodiff=true)

    @variable(model, -pi / 2 <= mie_coeffs[1:(2 * nu_max)] <= pi / 2)
    @NLobjective(model, Min, objective(mie_coeffs...))

    if method == "local"
        set_optimizer(model, Ipopt.Optimizer)
    elseif method == "global"
        local_opt = NLopt.Opt(:LD_MMA, 2nu_max)
        local_opt.maxeval = 500
        local_opt.ftol_rel = 1e-9
        local_opt.xtol_rel = 1e-6
        set_optimizer(model, NLopt.Optimizer)
        set_optimizer_attributes(
            model,
            "algorithm" => :GD_MLSL_LDS,
            "population" => 2^ndims(local_opt),
            "maxeval" => 10 * local_opt.maxeval * 2^ndims(local_opt),
            "local_optimizer" => local_opt,
        )
    end

    println(args)
    println(model)
    JuMP.optimize!(model)
    println(solution_summary(model; verbose=true))
    return nothing
end

main()
