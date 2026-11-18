#include "Prescribed.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace driver {

Prescribed::Prescribed(const YAML::Node& input, KynemaFMB& target)
    : m_target(target)
    , m_dt(target.get_timestep_size())
    , m_forces(target.num_nodes())
{
    const auto loads = input["loads"];
    if (!loads || !loads.IsSequence() || loads.size() == 0)
        throw std::runtime_error("Prescribed loads must be a nonempty sequence");
    for (const auto& entry : loads) {
        const int node = entry["node"].as<int>();
        const int step = entry["start_step"].as<int>();
        const auto force = entry["force"].as<std::array<double, 6>>();
        if (node < 0 || static_cast<size_t>(node) >= target.num_nodes())
            throw std::runtime_error("Prescribed load node is out of range");
        if (step < 1)
            throw std::runtime_error("Prescribed start_step must be positive (one-based)");
        for (double value : force)
            if (!std::isfinite(value))
                throw std::runtime_error("Prescribed forces must be finite");
        for (const auto& load : m_loads)
            if (load.node == static_cast<size_t>(node) && load.start_step == step)
                throw std::runtime_error("Duplicate prescribed node/start_step");
        m_loads.push_back({step, static_cast<size_t>(node), force});
    }
    std::sort(m_loads.begin(), m_loads.end(), [](const Load& a, const Load& b) {
        return a.start_step < b.start_step;
    });
}

void Prescribed::set_inputs(size_t)
{
    std::fill(m_forces.begin(), m_forces.end(), std::array<double, 6>{});
    for (const auto& load : m_loads) {
        if (load.start_step > m_step + 1) break;
        m_forces[load.node] = load.force;
    }
    m_target.set_forces(m_forces);
}

void Prescribed::set_timestep_size(double dt)
{
    if (dt != m_dt)
        throw std::runtime_error("Prescribed module requires its target FMB timestep");
}

void Prescribed::finalize_time_step()
{
    ++m_step;
    m_time += m_dt;
}

} // namespace driver
