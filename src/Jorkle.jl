module Jorkle

export backscattered_energy, total_energy, directivity, get_lm

using AssociatedLegendrePolynomials
using SpecialFunctions
using Integrals
using QuadGK
using Memoization

const c_0 = 2.99792458e8
const Z_0 = 376.730313668

function norm_param(l, m)
    if abs(m) > l
        return 0.0
    end
    return 1im^float(m) *
           sqrt((2l + 1) / (4π * l * (l + 1))) *
           sqrt(exp(logabsgamma(l - m + 1)[1] - logabsgamma(l + m + 1)[1]))
end

function legendre(m, l, theta)
    if abs(m) > l
        return 0.0
    end
    a = m < 0 ? (-1)^m * factorial(l + m) / factorial(l - m) : 1
    return a * Plm(l, abs(m), cos(theta))
end

@memoize function tau(m, l, h, theta)
    lmn = legendre(m, l, real(theta))
    tau_2 = legendre(m + 1, l, real(theta)) + m * cos(theta) / sin(theta) * lmn
    tau_1 = m * lmn / sin(theta)
    return -tau_2 - h * tau_1
end

function get_lm(l_max)
    l = vcat([fill(i, 2i + 1) for i in 1:l_max]...)
    m = vcat([(-i):i for i in 1:l_max]...)
    return l, m
end

function theta_0(wi, wp, b, g)
    z = (wi - g * wp) / b / g / wp

    return real(-1im * log(z + sqrt(complex(z^2 - 1))))
end

function heaviside(x, y)
    return ifelse(x < 0, zero(x), ifelse(x > 0, one(x), oftype(x, y)))
end

function gaussian_profile(hi, theta, phi, wi, w0)
    return (
        sin(2theta) *
        exp(-(wi^2) * w0^2 * sin(theta)^2 / (4c_0^2)) *
        heaviside(π / 2 - theta, 1) *
        exp(1im * hi * phi)
    )
end

function theta_rtp(Tinc, theta, phi)
    z = sin(theta) * cos(phi) * sin(Tinc) + cos(theta) * cos(Tinc)
    return real(-1im * log(z + sqrt(complex(z^2 - 1))))
end

function phi_rtp(Tinc, theta, phi)
    x = sin(theta) * cos(phi) * cos(Tinc) - cos(theta) * sin(Tinc)
    y = sin(theta) * sin(phi)

    a = y / x

    return real(0.5 * 1im * log((1 - a * 1im) / (1 + a * 1im)))
end

function jacobian_rotated(theta, phi, Tinc)
    return (
        sin(theta) *
        (
            sin(phi)^2 * sin(theta)^2 +
            (cos(phi) * cos(Tinc) * sin(theta) + (-1) * cos(theta) * sin(Tinc))^2
        )^(-1) *
        (
            sin(phi)^2 * sin(Tinc)^2 +
            (cos(Tinc) * sin(theta) + (-1) * cos(phi) * cos(theta) * sin(Tinc))^2
        ) *
        (1 + (-1) * (cos(theta) * cos(Tinc) + cos(phi) * sin(theta) * sin(Tinc))^2)^(-1 / 2)
    )
end

function phase(t, p, T, h, tp, pp)
    return (
        (1 / 2) * (
            (cos(p) + 1im * h * cos(t) * sin(p)) *
            (cos(pp) + (1im * (-1)) * h * cos(tp) * sin(pp)) +
            h *
            sin(t) *
            (
                h * cos(pp) * cos(tp) * sin(T) +
                (1im * (-1)) * sin(pp) * sin(T) +
                h * cos(T) * sin(tp)
            ) +
            (h * cos(p) * cos(t) + 1im * sin(p)) * (
                h * cos(pp) * cos(T) * cos(tp) +
                (1im * (-1)) * cos(T) * sin(pp) +
                (-1) * h * sin(T) * sin(tp)
            )
        )
    )
end

function g_coeff(theta, phi, hi, wi, w0, Tinc)
    tbd = theta_rtp(Tinc, theta, phi)
    pbd = phi_rtp(Tinc, theta, phi)

    gauss = gaussian_profile(hi, tbd, pbd, wi, w0)

    J = jacobian_rotated(theta, phi, Tinc)

    P = phase(theta, phi, Tinc, hi, tbd, pbd)

    return gauss * J * P
end

function s_coeff(theta, phi, l, m, hi)
    pf = 4 * pi * 1im^float(l + 2 * m - 1) * norm_param(l, m)

    t = tau(m, l, hi, theta)

    exponential = exp(-1im * m * phi)

    return pf * t * exponential
end

function theta_btr(t, b)
    z = (cos(t) + b) / (1 + b * cos(t))

    return real(-1im * log(z + sqrt(complex(z^2 - 1))))
end

function phi_btr(p)
    return p
end

function A_coeff(phi, wp, b, g, Tinc, wi, hi, w0, l, m)
    t0 = theta_0(wi, wp, b, g)

    denom = b * g^2 * wp * sin(t0) * (1 + b * cos(t0))

    t_btr = theta_btr(t0, b)
    p_btr = phi_btr(phi)

    S = s_coeff(t0, phi, l, m, hi)

    G = g_coeff(t_btr, p_btr, hi, wi, w0, Tinc)

    return G * S / denom
end

@memoize function I(m, l, lp, hr, g, b; atol, rtol)
    prob = IntegralProblem(0.01, pi-0.01) do theta, _
        theta_p_sca = acos((cos(theta) - b) / (1 - b * cos(theta)))
        doppler = (g * (1 + b * cos(theta_p_sca)))^3
        return (
            sin(theta) *
            doppler *
            tau(m, l, hr, theta_p_sca) *
            tau(m, lp, hr, theta_p_sca)
        )
    end
    return only(solve(prob, QuadGKJL(); abstol=atol, reltol=rtol))
end

@memoize function I_B(b, Tinc, hi, wi, w0, m, mp, l, lp, g; atol, rtol)
    c = g * (1 - b * cos(Tinc))
    outer = IntegralProblem(wi/g/(1+b), c, wi/g/(1-b)) do wp, _
        inner = IntegralProblem(0.0, 2pi) do phi, _
            return (
                1 / wp^2 *
                A_coeff(phi, wp, b, g, Tinc, wi, hi, w0, l, m) *
                conj(A_coeff(phi, wp, b, g, Tinc, wi, hi, w0, lp, mp))
            )
        end
        only(solve(inner, QuadGKJL(); abstol=atol, reltol=rtol))
    end
    return only(solve(outer, QuadGKJL(); abstol=10 * atol, reltol=10 * rtol))
end

function get_tmatrix(mie_elec, mie_mag, hi, hr)
    alpha = π / 2 .- mie_elec
    b = π / 2 .- mie_mag
    a = @. -1im * sin(alpha) * exp(-1im * alpha)
    b = @. -1im * sin(b) * exp(-1im * b)
    return (a + hi * hr * b) / 2
end

function backscattered_energy(
    mie_elec,
    mie_mag,
    theta,
    phi,
    hi,
    hr,
    wi,
    w0,
    g,
    b,
    m,
    l,
    atol=eps(Float64),
    rtol=sqrt(eps(Float64)),
)
    T = get_tmatrix(mie_elec, mie_mag, hi, hr)

    integral = 0.0
    pp = 4pi * c_0^2 / Z_0
    phi_sca = pi + phi
    theta_bs = pi - theta
    theta_p_sca = acos((cos(theta_bs) - b) / (1 - b * cos(theta_bs)))
    doppler = (g * (1 + b * cos(theta_p_sca)))^3

    ml = Iterators.product(map(unique, [m, m, l, l])...)

    for (mi, mpi, li, lpi) in ml
        prefactor =
            pp *
            doppler *
            (-1.0im)^(li - lpi) *
            norm_param(li, mi) *
            conj(norm_param(lpi, mpi)) *
            tau(mi, li, hr, theta_p_sca) *
            conj(tau(mpi, lpi, hr, theta_p_sca)) *
            exp(1im * (mi - mpi) * phi_sca)
        integral += real(
            prefactor *
            conj(T[lpi]) *
            T[li] *
            I_B(b, theta, hi, wi, w0, mi, mpi, li, lpi, g; atol=atol, rtol=rtol),
        )
    end
    return integral
end

function total_energy(
    mie_elec,
    mie_mag,
    theta,
    hi,
    hr,
    wi,
    w0,
    g,
    b,
    m,
    l,
    atol=eps(Float64),
    rtol=sqrt(eps(Float64)),
)
    T = get_tmatrix(mie_elec, mie_mag, hi, hr)

    ml = Iterators.filter(Iterators.product(map(unique, [m, l, l])...)) do x
        (x[2] <= x[3]) & (abs(x[1]) <= abs(x[3])) & (abs(x[1]) <= abs(x[2]))
    end

    integral = 0
    pp = 8pi^2 * c_0^2 / Z_0
    for (mi, li, lpi) in ml
        fac = (li == lpi) ? 1.0 : 2.0
        prefactor =
            fac * pp * (-1.0im)^(lpi - li) * conj(norm_param(li, mi)) * norm_param(lpi, mi)
        integral += real(
            prefactor *
            conj(T[li]) *
            T[lpi] *
            I_B(b, theta, hi, wi, w0, mi, mi, li, lpi, g; atol=atol, rtol=rtol) *
            I(mi, li, lpi, hr, g, b; atol=atol, rtol=rtol),
        )
    end
    return integral
end

function directivity(
    mie_elec,
    mie_mag,
    theta,
    phi,
    h_inc,
    w_i,
    w_0,
    gamma,
    beta,
    m,
    l,
    atol=eps(Float64),
    rtol=sqrt(eps(Float64)),
)
    bs_p = backscattered_energy(
        mie_elec, mie_mag, theta, phi, h_inc, 1, w_i, w_0, gamma, beta, m, l, atol, rtol
    )
    bs_m = backscattered_energy(
        mie_elec, mie_mag, theta, phi, h_inc, -1, w_i, w_0, gamma, beta, m, l, atol, rtol
    )
    tot_p = total_energy(
        mie_elec, mie_mag, theta, h_inc, 1, w_i, w_0, gamma, beta, m, l, atol, rtol
    )
    tot_m = total_energy(
        mie_elec, mie_mag, theta, h_inc, -1, w_i, w_0, gamma, beta, m, l, atol, rtol
    )
    return 4pi * (bs_p + bs_m) / (tot_p + tot_m)
end

end
