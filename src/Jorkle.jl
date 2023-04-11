module Jorkle

export backscattered_energy, total_energy, directivity, get_munu

using AssociatedLegendrePolynomials
using SpecialFunctions
using Integrals
using QuadGK
using Memoization

const c_0 = 2.99792458e8
const Z_0 = 376.730313668

function norm_param(m, n)
    if abs(m) > n
        return 0.0
    end
    return 1im^float(m) *
           sqrt((2n + 1) / (4π * n * (n + 1))) *
           sqrt(exp(logabsgamma(n - m + 1)[1] - logabsgamma(n + m + 1)[1]))
end

function legendre(m, n, theta)
    if abs(m) > n
        return 0.0
    end
    a = m < 0 ? (-1)^m * factorial(n + m) / factorial(n - m) : 1
    return a * Plm(n, abs(m), cos(theta))
end

@memoize function tau(m, n, h, theta)
    lmn = legendre(m, n, real(theta))
    tau_1 = legendre(m + 1, n, real(theta)) + m * cos(theta) / sin(theta) * lmn
    tau_2 = m * lmn / sin(theta)
    return -tau_1 - h * tau_2
end

function get_munu(n)
    mu = vcat([(-i):i for i in 1:n]...)
    nu = vcat([fill(i, 2i + 1) for i in 1:n]...)
    return mu, nu
end

function theta_root(beta, omega_i, omega_p, gamma)
    a = (gamma * omega_i - beta^2 * gamma * omega_i - omega_p) / beta / omega_p
    return -1im * log(a + 1im * sqrt(complex(1 - a^2)))
end

function theta_func(theta_p, phi_p, theta, beta, gamma)
    a = (
        (
            sin(theta_p) * cos(phi_p) * sin(theta) +
            gamma * (beta + cos(theta_p)) * cos(theta)
        ) / (gamma * (1 + beta * cos(theta_p)))
    )
    return real(-1im * log(a + 1im * sqrt(1 - a^2)))
end

function phi_func(theta_p, phi_p, theta, beta, gamma)
    x = sin(theta_p) * cos(phi_p) * cos(theta) - gamma * (cos(theta_p) + beta) * sin(theta)
    y = sin(theta_p) * sin(phi_p)
    return (real(-1im * log((x + 1im * y) / sqrt(x^2 + y^2))))
end

function diff_theta_func(beta, omega_i, theta_0, gamma)
    return abs(
        beta * (beta^2 - 1) * gamma * omega_i * sin(theta_0) / (1 + beta * cos(theta_0))^2
    )
end

function jacobian_func(beta, theta_p, phi_p, theta, gamma)
    stp = sin(theta_p)
    ctp = cos(theta_p)
    st = sin(theta)
    ct = cos(theta)
    spp = sin(phi_p)
    cpp = cos(phi_p)

    a = gamma * ct * (beta + ctp) + cpp * st * stp
    b = cpp * ct * stp
    c = -gamma * (beta + ctp) * st + b
    d = (spp * stp)^2 + c^2
    e = 1 + beta * ctp

    return (
        (
            stp * (
                -c * ctp * e * st +
                c * cpp * (-a * beta + ct * e * gamma) * stp +
                spp^2 * (-a * beta * ct + e * gamma) * stp^2
            )
        ) / (d * e^2 * sqrt(1 - (a / (e * gamma))^2) * gamma)
    )
end

function heaviside(x, y)
    return ifelse(x < 0, zero(x), ifelse(x > 0, one(x), oftype(x, y)))
end

function gaussian_profile(theta, omega_i, omega_0)
    return (
        sin(2theta) *
        exp(-(omega_i^2) * omega_0^2 * sin(theta)^2 / (4c_0^2)) *
        heaviside(π / 2 - theta, 1)
    )
end

function A_coeff(phi_p, omega_p, beta, theta, h_inc, w_i, w_0, mu, nu, gamma)
    theta_0 = theta_root(beta, w_i, omega_p, gamma)
    theta_A = theta_func(theta_0, phi_p, theta, beta, gamma)

    phi_A = phi_func(theta_0, phi_p, theta, beta, gamma)

    a = tau(mu, nu, h_inc, theta_0) * exp(-1im * mu * phi_p)

    phase = (
        (1 / 2) * (
            (1 + h_inc^2) *
            (cos(theta) * sin(theta_A) + cos(theta_A) * cos(phi_A) * sin(theta)) +
            -2im * h_inc * sin(theta) * sin(phi_A)
        ) / sqrt(
            (cos(theta) * cos(phi_A) * sin(theta_A) + cos(theta_A) * sin(theta))^2 +
            sin(theta_A)^2 * sin(phi_A)^2,
        )
    )

    b = (
        gamma *
        (1 - beta * (cos(theta_0) + beta) / (1 + beta * cos(theta_0))) *
        jacobian_func(beta, theta_0, phi_p, theta, gamma) *
        gaussian_profile(theta_A, w_i, w_0) / diff_theta_func(beta, w_i, theta_0, gamma)
    )

    return a * b * phase
end

@memoize function I(mu, nu, nu_p, h_rad, gamma, beta; atol, rtol)
    prob = IntegralProblem(0.0, pi) do theta, _
        theta_p_sca = acos((cos(theta) - beta) / (1 - beta * cos(theta)))
        doppler = (gamma * (1 + beta * cos(theta_p_sca)))^3
        return (
            sin(theta_p_sca) *
            doppler *
            tau(mu, nu, h_rad, theta_p_sca) *
            tau(mu, nu_p, h_rad, theta_p_sca)
        )
    end
    return only(solve(prob, QuadGKJL(); abstol=atol, reltol=rtol))
end

@memoize function I_B(beta, theta, h_inc, w_i, w_0, mu, mu_p, nu, nu_p, gamma; atol, rtol)
    c = gamma * (1 - beta * cos(theta))
    outer = IntegralProblem(c - beta * c / 4, c + beta * c / 4) do omega, _
        inner = IntegralProblem(0.0, 2pi) do phi, _
            return (
                1 / omega^2 *
                A_coeff(phi, omega, beta, theta, h_inc, w_i, w_0, mu, nu, gamma) *
                conj(A_coeff(phi, omega, beta, theta, h_inc, w_i, w_0, mu_p, nu_p, gamma))
            )
        end
        only(solve(inner, QuadGKJL(); abstol=atol, reltol=rtol))
    end
    return only(solve(outer, QuadGKJL(); abstol=10 * atol, reltol=10 * rtol))
end

function get_tmatrix(mie_elec, mie_mag, h_inc, h_rad)
    alpha = π / 2 .- mie_elec
    beta = π / 2 .- mie_mag
    a = @. -1im * sin(alpha) * exp(-1im * alpha)
    b = @. -1im * sin(beta) * exp(-1im * beta)
    return (a + h_inc * h_rad * b) / 2
end

function backscattered_energy(
    mie_elec,
    mie_mag,
    theta,
    phi,
    h_inc,
    h_rad,
    w_i,
    w_0,
    gamma,
    beta,
    mu,
    nu,
    atol=eps(Float64),
    rtol=sqrt(eps(Float64)),
)
    T = get_tmatrix(mie_elec, mie_mag, h_inc, h_rad)

    integral = 0.0
    pp = 4pi * c_0^2 / Z_0
    phi_sca = pi + phi
    theta_p_sca = acos((cos(theta) - beta) / (1 - beta * cos(theta)))
    doppler = (gamma * (1 + beta * cos(theta_p_sca)))^3

    munu = Iterators.product(map(unique, [mu, mu, nu, nu])...)

    for (mui, mupi, nui, nupi) in munu
        prefactor =
            pp *
            doppler *
            (-1.0im)^(nui - nupi) *
            norm_param(mui, nui) *
            conj(norm_param(mupi, nupi)) *
            tau(mui, nui, h_rad, theta_p_sca) *
            conj(tau(mupi, nupi, h_rad, theta_p_sca)) *
            exp(1im * (mui - mupi) * phi_sca)
        integral += real(
            prefactor *
            conj(T[nupi]) *
            T[nui] *
            I_B(
                beta,
                theta,
                h_inc,
                w_i,
                w_0,
                mui,
                mupi,
                nui,
                nupi,
                gamma;
                atol=atol,
                rtol=rtol,
            ),
        )
    end
    return integral
end

function total_energy(
    mie_elec,
    mie_mag,
    theta,
    h_inc,
    h_rad,
    w_i,
    w_0,
    gamma,
    beta,
    mu,
    nu,
    atol=eps(Float64),
    rtol=sqrt(eps(Float64)),
)
    T = get_tmatrix(mie_elec, mie_mag, h_inc, h_rad)

    munu = Iterators.filter(Iterators.product(map(unique, [mu, nu, nu])...)) do x
        (x[2] <= x[3]) & (abs(x[1]) <= abs(x[3])) & (abs(x[1]) <= abs(x[2]))
    end

    integral = 0
    pp = 8pi^2 * c_0^2 / Z_0
    for (mui, nui, nupi) in munu
        fac = (nui == nupi) ? 1.0 : 2.0
        prefactor =
            fac *
            pp *
            (-1.0im)^(nupi - nui) *
            norm_param(mui, nui) *
            conj(norm_param(mui, nupi))
        integral += real(
            prefactor *
            conj(T[nupi]) *
            T[nui] *
            I_B(
                beta,
                theta,
                h_inc,
                w_i,
                w_0,
                mui,
                mui,
                nui,
                nupi,
                gamma;
                atol=atol,
                rtol=rtol,
            ) *
            I(mui, nui, nupi, h_rad, gamma, beta; atol=atol, rtol=rtol),
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
    mu,
    nu,
    atol=eps(Float64),
    rtol=sqrt(eps(Float64)),
)
    bs_p = backscattered_energy(
        mie_elec, mie_mag, theta, phi, h_inc, 1, w_i, w_0, gamma, beta, mu, nu, atol, rtol
    )
    bs_m = backscattered_energy(
        mie_elec, mie_mag, theta, phi, h_inc, -1, w_i, w_0, gamma, beta, mu, nu, atol, rtol
    )
    tot_p = total_energy(
        mie_elec, mie_mag, theta, h_inc, 1, w_i, w_0, gamma, beta, mu, nu, atol, rtol
    )
    tot_m = total_energy(
        mie_elec, mie_mag, theta, h_inc, -1, w_i, w_0, gamma, beta, mu, nu, atol, rtol
    )
    return 4pi * (bs_p + bs_m) / (tot_p + tot_m)
end

end
