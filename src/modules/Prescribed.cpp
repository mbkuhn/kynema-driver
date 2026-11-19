#include "Prescribed.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace driver {

bool PrescribedSource::prepare()
{
    if (m_initialization.is_null()) return false;
    m_name += "-" + m_initialization.at("provider").get<std::string>();
    m_dt = m_initialization.at("time_step").get<double>();
    m_time = m_initialization.at("time").get<double>();
    m_step = m_initialization.at("step").get<int>();
    m_field = m_initialization.at("field").get<std::string>();
    m_version = m_initialization.at("version").get<int>();
    const auto items = m_initialization.at("items").get<size_t>();
    const auto components = m_initialization.at("components").get<size_t>();
    if (!std::isfinite(m_dt) || m_dt <= 0. || !std::isfinite(m_time) ||
        m_step < 0 || m_field.empty() || m_field == "version" ||
        items == 0 || components == 0)
        throw std::invalid_argument("Invalid prescribed initialization data");

    const auto loads = m_input["loads"];
    if (!loads || !loads.IsSequence() || loads.size() == 0)
        throw std::runtime_error("Prescribed loads must be a nonempty sequence");
    for (const auto& entry : loads) {
        const int node = entry["node"].as<int>();
        const int step = entry["start_step"].as<int>();
        const auto force = entry["force"].as<std::vector<double>>();
        if (node < 0 || static_cast<size_t>(node) >= items)
            throw std::runtime_error("Prescribed load node is out of range");
        if (step < 1)
            throw std::runtime_error("Prescribed start_step must be positive (one-based)");
        if (force.size() != components)
            throw std::runtime_error("Prescribed force component count does not match the target");
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
    m_values.assign(items, std::vector<double>(components));
    return true;
}

nlohmann::json PrescribedSource::outputs() const
{
    return {{"version", m_version}, {m_field, m_values}};
}

void PrescribedSource::calculate_outputs(std::size_t)
{
    for (auto& values : m_values) std::fill(values.begin(), values.end(), 0.);
    for (const auto& load : m_loads) {
        if (load.start_step > m_step + 1) break;
        m_values[load.node] = load.force;
    }
}

void PrescribedSource::set_timestep_size(double dt)
{
    if (dt != m_dt)
        throw std::runtime_error("Prescribed module requires its target timestep");
}

void PrescribedSource::finalize_time_step()
{
    ++m_step;
    m_time += m_dt;
}

} // namespace driver
