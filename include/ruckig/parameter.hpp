#pragma once

#include <optional>

#include <Eigen/Core>


namespace ruckig {

enum class Result {
    Working,
    Finished,
    Error
};


template<size_t DOFs>
struct InputParameter {
    using Vector = Eigen::Matrix<double, DOFs, 1, Eigen::ColMajor>;
    static constexpr size_t degrees_of_freedom {DOFs};

    enum class Type {
        Position,
        Velocity,
    } type {Type::Position};

    Vector current_position;
    Vector current_velocity {Vector::Zero()};
    Vector current_acceleration {Vector::Zero()};

    Vector target_position;
    Vector target_velocity {Vector::Zero()};
    Vector target_acceleration {Vector::Zero()};

    Vector max_velocity;
    Vector max_acceleration;
    Vector max_jerk;

    std::array<bool, DOFs> enabled;
    std::optional<double> minimum_duration;

    InputParameter() {
        enabled.fill(true);
    }

    bool operator!=(const InputParameter<DOFs>& rhs) const {
        return (
            current_position != rhs.current_position
            || current_velocity != rhs.current_velocity
            || current_acceleration != rhs.current_acceleration
            || target_position != rhs.target_position
            || target_velocity != rhs.target_velocity
            || target_acceleration != rhs.target_acceleration
            || max_velocity != rhs.max_velocity
            || max_acceleration != rhs.max_acceleration
            || max_jerk != rhs.max_jerk
            || enabled != rhs.enabled
            || minimum_duration != rhs.minimum_duration
            || type != rhs.type
        );
    }

    std::string to_string(size_t dof) const {
        std::stringstream ss;
        ss << "p0: " << current_position[dof] << ", ";
        ss << "v0: " << current_velocity[dof] << ", ";
        ss << "a0: " << current_acceleration[dof] << ", ";
        ss << "pf: " << target_position[dof] << ", ";
        ss << "vf: " << target_velocity[dof] << ", ";
        ss << "vMax: " << max_velocity[dof] << ", ";
        ss << "aMax: " << max_acceleration[dof] << ", ";
        ss << "jMax: " << max_jerk[dof];
        return ss.str();
    }

    std::string to_string() const {
        Eigen::IOFormat formatter(10, 0, ", ", "\n", "[", "]");

        std::stringstream ss;
        ss << "\ninp.current_position = " << current_position.transpose().format(formatter) << "\n";
        ss << "inp.current_velocity = " << current_velocity.transpose().format(formatter) << "\n";
        ss << "inp.current_acceleration = " << current_acceleration.transpose().format(formatter) << "\n";
        ss << "inp.target_position = " << target_position.transpose().format(formatter) << "\n";
        ss << "inp.target_velocity = " << target_velocity.transpose().format(formatter) << "\n";
        ss << "inp.target_acceleration = " << target_acceleration.transpose().format(formatter) << "\n";
        ss << "inp.max_velocity = " << max_velocity.transpose().format(formatter) << "\n";
        ss << "inp.max_acceleration = " << max_acceleration.transpose().format(formatter) << "\n";
        ss << "inp.max_jerk = " << max_jerk.transpose().format(formatter) << "\n";
        return ss.str();
    }
};


template<size_t DOFs>
struct OutputParameter {
    using Vector = Eigen::Matrix<double, DOFs, 1, Eigen::ColMajor>;
    static constexpr size_t degrees_of_freedom {DOFs};

    Vector new_position;
    Vector new_velocity;
    Vector new_acceleration;

    double duration; // [s]
    bool new_calculation {false};
    double calculation_duration; // [µs]

    // Vector independent_min_durations;
    // Vector min_positions, max_positions;
    // Vector time_min_positions, time_max_positions;
};

} // namespace ruckig
