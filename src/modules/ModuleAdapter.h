#ifndef DRIVER_MODULE_ADAPTER_H
#define DRIVER_MODULE_ADAPTER_H

#include <utility>

#include "../Module.h"

namespace driver {

template <typename NativeModule>
struct ModuleAdapterTraits;

template <typename NativeModule>
class ModuleAdapter : public Module
{
    using Traits = ModuleAdapterTraits<NativeModule>;

public:
    template <typename... Args>
    explicit ModuleAdapter(Args&&... args)
        : m_native(std::forward<Args>(args)...)
    {}

    PreparationStage prepare(const PreparationContext& context) override
    {
        return Traits::prepare(m_native, context);
    }

    nlohmann::json initialization_data() const override
    {
        return m_native.initialization_data();
    }
    void receive_initialization_data(const nlohmann::json& data) override
    {
        m_native.receive_initialization_data(data);
    }
    void receive_inputs(const nlohmann::json& inputs) override
    {
        m_native.receive_inputs(inputs);
    }
    nlohmann::json outputs() const override { return m_native.outputs(); }

    std::string identifier() override { return m_native.identifier(); }
    bool is_fixed_timestep_size() override
    {
        return Traits::is_fixed_timestep_size(m_native);
    }
    int time_index() override { return m_native.time_index(); }
    double get_time() override { return m_native.get_time(); }
    double get_timestep_size() override { return m_native.get_timestep_size(); }
    void set_timestep_size(double dt) override
    {
        m_native.set_timestep_size(dt);
    }

    void set_inputs(std::size_t iteration) override
    {
        m_native.set_inputs(iteration);
    }
    void predict_inputs(std::size_t iteration) override
    {
        m_native.predict_inputs(iteration);
    }
    void reset_states() override { m_native.reset_states(); }
    void advance_states(std::size_t iteration) override
    {
        m_native.advance_states(iteration);
    }
    void calculate_outputs(std::size_t iteration) override
    {
        m_native.calculate_outputs(iteration);
    }
    void additional_picard_iterations(int count) override
    {
        m_native.additional_picard_iterations(count);
    }
    void finalize_time_step() override { m_native.finalize_time_step(); }
    void dump_simulation_time() override { m_native.dump_simulation_time(); }

protected:
    NativeModule& native() { return m_native; }

private:
    NativeModule m_native;
};

} // namespace driver
#endif
