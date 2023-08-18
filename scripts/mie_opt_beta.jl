using ArgParse
using JuMP
using Ipopt
using NLopt
using DelimitedFiles

using Jorkle
import Jorkle: c_0

function cmdline_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--l_max"
        arg_type = Int64
        default = 2
        "--theta"
        arg_type = Float64
        default = pi / 4
        "--n_beta"
        arg_type = Int64
        default = 10
        "--start_beta"
        arg_type = Float64
        default = 0.1
        "--end_beta"
        arg_type = Float64
        default = 0.9
    end
    return parse_args(s)
end

function main()

    args = cmdline_args()

    l_max = args["l_max"]
    theta = args["theta"]
    n_beta = args["n_beta"]
    start_beta = args["start_beta"]
    end_beta = args["end_beta"]

    h_inc = 1
    phi = 0.0

    wi = 1.0

    w0_ratio = 10.0
    w0 = w0_ratio * 2pi * c_0 / wi

    l, m = get_lm(l_max)

    atol = eps(Float64)
    rtol = sqrt(eps(Float64))
    

    beta = LinRange(start_beta, end_beta, n_beta)
    #beta = reverse(LinRange(start_beta, end_beta, n_beta)) # something like 0.1 to 0.9
    start = [3.28985e-01, 1.06966e+00, 1.43841e+00, 3.20618e-01, 1.06325e+00, 1.43194e+00] # initial Mie coefficients from first optimisation

    mie_opt = Array{Float64}(undef, (2l_max, n_beta))
    obj = Array{Float64}(undef, (1, n_beta)) # Objective function of corresponding Mie angle

    for i = 1 : n_beta

        #optimizer = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-8)
    
        model = Model(Ipopt.Optimizer)
    
        @variable(model, mie_coeffs[1:(2 * l_max)])

        gamma = 1 / sqrt(1 - beta[i]^2)

        function objective(x...)
            s = length(x) / 2
            me = vcat(x[1:s]...)
            mm = vcat(x[(s + 1):end]...)
            d = directivity(me, mm, theta, phi, h_inc, wi, w0, gamma, beta[i], m, l, atol, rtol) 
            return d
        end

        register(model, :objective, 2l_max, objective; autodiff=true)

        delta = abs.(start / 1000)

        for j = 1 : 2l_max
            set_upper_bound(mie_coeffs[j], start[j] + delta[j])
            #set_upper_bound(mie_coeffs[j], pi/2)
            #println("upper", " ", "bound", " ", "is", " ", upper_bound(mie_coeffs[i]))
            set_lower_bound(mie_coeffs[j], start[j] - delta[j])
            #set_lower_bound(mie_coeffs[j], -pi/2)
            #println("lower", " ", "bound", " ", "is", " ", lower_bound(mie_coeffs[i]))
        end

        println(model)

        #println(mie_coeffs)

        @NLobjective(model, Min, objective(mie_coeffs...))

        JuMP.optimize!(model)

        start = JuMP.value.(mie_coeffs)

        obj_val = objective_value(model)

        mie_opt[:, i] = start
        obj[i] = obj_val

        delete(model, mie_coeffs)

        unregister(model, :objective)

        println(i)

    end

    writedlm("opt_mie_beta_dual_02_099.csv", mie_opt, ',')
    writedlm("obj_beta_dual_02_099.csv", obj, ',')

    return nothing
    
end

@time main()