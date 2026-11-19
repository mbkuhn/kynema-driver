#ifndef DRIVER_PRESCRIBED_H
#define DRIVER_PRESCRIBED_H

#include <vector>
#include <yaml-cpp/yaml.h>

#include "ModuleAdapter.h"

namespace driver {

class PrescribedSource
{
public:
    explicit PrescribedSource(const YAML::Node& input) : m_input(input) {}
    int time_index() const { return m_step; }
    std::string identifier() const { return m_name; }
    nlohmann::json outputs() const;

    bool prepare();
    void receive_initialization_data(const nlohmann::json& data)
    {
        m_initialization = data;
    }
    double get_time() const { return m_time; }
    double get_timestep_size() const { return m_dt; }
    void set_timestep_size(double dt);
    void set_inputs(std::size_t) {}
    void predict_inputs(std::size_t) {}
    void reset_states() {}
    void advance_states(std::size_t) {}
    void calculate_outputs(std::size_t);
    void additional_picard_iterations(int) {}
    void finalize_time_step();
    void dump_simulation_time() {}

private:
    struct Load
    {
        int start_step;
        size_t node;
        std::vector<double> force;
    };

    YAML::Node m_input;
    nlohmann::json m_initialization;
    std::string m_name{"Prescribed"};
    std::string m_field;
    int m_version{0};
    double m_dt{0.};
    double m_time{0.};
    int m_step{0};
    std::vector<Load> m_loads;
    std::vector<std::vector<double>> m_values;
};

template <>
struct ModuleAdapterTraits<PrescribedSource>
{
    static constexpr bool fixed_timestep = true;
    static constexpr bool publishes_initialization_data = false;
    static constexpr bool accepts_initialization_data = true;
    static constexpr bool accepts_inputs = false;

    static PreparationStage
    prepare(PrescribedSource& source, const PreparationContext& context)
    {
        return source.prepare() ? PreparationStage::complete : context.stage;
    }
};

using Prescribed = ModuleAdapter<PrescribedSource>;

} // namespace driver
#endif
