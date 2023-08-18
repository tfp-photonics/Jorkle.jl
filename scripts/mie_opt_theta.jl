using ArgParse
using JuMP
using Ipopt
using DelimitedFiles

using Jorkle
import Jorkle: c_0

function cmdline_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--l_max"
        arg_type = Int64
        default = 2
        "--beta"
        arg_type = Float64
        default = 0.2
        "--n_theta"
        arg_type = Int64
        default = 10
        "--start_theta"
        arg_type = Float64
        default = pi/4
        "--end_theta"
        arg_type = Float64
        default = pi-0.01
    end
    return parse_args(s)
end

function main()

    args = cmdline_args()

    l_max = args["l_max"]
    beta = args["beta"]
    n_theta = args["n_theta"]
    start_theta = args["start_theta"]
    end_theta = args["end_theta"]

    h_inc = 1
    phi = 0.0

    wi = 1.0

    w0_ratio = 10.0
    w0 = w0_ratio * 2pi * c_0 / wi

    l, m = get_lm(l_max)

    atol = eps(Float64)
    rtol = 1e-10

    theta = LinRange(start_theta, end_theta, n_theta) 
    start = [0.328985,1.06966,1.43841,0.320618,1.06325,1.43194] # initial Mie coefficients from first optimisation

    mie_opt = Array{Float64}(undef, (2l_max, n_theta))
    obj = Array{Float64}(undef, (1, n_theta)) # Objective function of corresponding Mie angle

    for i = 1 : n_theta

        model = Model(Ipopt.Optimizer)
    
        @variable(model, mie_coeffs[1:(2 * l_max)])

        gamma = 1 / sqrt(1 - beta^2)

        function objective(x...)
            s = length(x) ÷ 2
            me = vcat(x[1:s]...)
            mm = vcat(x[(s + 1):end]...)
            d = directivity(me, mm, theta[i], phi, h_inc, wi, w0, gamma, beta, m, l, atol, rtol) 
            return d
        end

        register(model, :objective, 2l_max, objective; autodiff=true)


        delta = abs.(start / (1000))


        for j = 1 : 2l_max
            set_upper_bound(mie_coeffs[j], start[j] + delta[j])
            set_lower_bound(mie_coeffs[j], start[j] - delta[j])
        
        end

        println(model)


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

    writedlm("100_opt_mie_dual_theta.csv", mie_opt, ',')
    writedlm("100_obj_theta_dual.csv", obj, ',')

    return nothing
    
end

@time main()