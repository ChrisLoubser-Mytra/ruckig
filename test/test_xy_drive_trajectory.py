# This example shows the usage of intermediate waypoints. It will only work with Ruckig Pro or enabled cloud API (e.g. default when installed by pip / PyPI).

from copy import copy
from dataclasses import dataclass, field
from typing import Tuple

from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory
import numpy as np

import plotly.graph_objects as go

@dataclass
class PositionVelocityAcceleration:
    position: float
    velocity: float
    acceleration: float

@dataclass
class PositionVelocityAccelerationHistory:
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    accelerations: list[float] = field(default_factory=list)
    

@dataclass
class MockMotorConfig:
    velocity_jitter : float = 0.01 # m/s
    acceleration: float | None = None # m/s^2
    time_step: float = 0.01 # seconds
    
@dataclass
class MockMotor:
    mock_motor_config: MockMotorConfig = field(default_factory=MockMotorConfig)
    current: PositionVelocityAcceleration = field(default_factory=lambda: PositionVelocityAcceleration(0, 0, 0))
    previous: PositionVelocityAcceleration = field(default_factory=lambda: PositionVelocityAcceleration(0, 0, 0))
    history: PositionVelocityAccelerationHistory = field(default_factory=PositionVelocityAccelerationHistory)
    
    def save_history(self):
        self.previous.position = self.current.position
        self.previous.velocity = self.current.velocity
        self.previous.acceleration = self.current.acceleration
        self.history.positions.append(self.current.position)
        self.history.velocities.append(self.current.velocity)
        self.history.accelerations.append(self.current.acceleration)
    
    def process_command(self, target_velocity: float, target_acceleration: float):
        """
        Process the command by calculating the new position, velocity, and acceleration
        Add some jitter to the velocity
        """
        self.save_history()
        self.current = self.calculate_position_velocity_acceleration(target_velocity, target_acceleration, self.mock_motor_config.velocity_jitter)
        
        
    def calculate_position_velocity_acceleration(self, target_velocity: float, target_acceleration: float, velocity_jitter: float = 0.0) -> PositionVelocityAcceleration:
        if target_velocity > self.current.velocity:
            acceleration = self.mock_motor_config.acceleration if self.mock_motor_config.acceleration is not None else target_acceleration
        elif target_velocity < self.current.velocity:
            acceleration = -self.mock_motor_config.acceleration if self.mock_motor_config.acceleration is not None else target_acceleration
        else:
            acceleration = 0
        
        jitter = np.random.uniform(-velocity_jitter, velocity_jitter)
        additional_velocity = acceleration * self.mock_motor_config.time_step + jitter
        velocity = self.current.velocity + additional_velocity
        # Calculate the current position
        position = self.current.position + (self.previous.velocity * self.mock_motor_config.time_step) + (0.5 * acceleration * self.mock_motor_config.time_step ** 2)
        
        return PositionVelocityAcceleration(position, velocity, acceleration)
        

def generate_offline_trajectory(inp : InputParameter, timestep :float) -> PositionVelocityAccelerationHistory:
    # Calculate entire trajectory at once
    offline_input : InputParameter = InputParameter(1)  # DoFs
    offline_input.current_position = inp.current_position
    offline_input.current_velocity = inp.current_velocity
    offline_input.current_acceleration = inp.current_acceleration
    offline_input.target_position = inp.target_position
    offline_input.target_velocity = inp.target_velocity
    offline_input.target_acceleration = inp.target_acceleration
    offline_input.max_velocity = inp.max_velocity
    offline_input.max_acceleration = inp.max_acceleration
    offline_ruckig = Ruckig(1)  # DoFs, control cycle rate
    offline_trajectory = Trajectory(1)
    offline_result = offline_ruckig.calculate(offline_input, offline_trajectory)
    if offline_result == Result.ErrorInvalidInput:
        raise Exception('Invalid input!')
    offline_positions :list[float]= []
    offline_velocities :list[float]= []
    offline_accelerations :list[float]= []
    for t in np.arange(0, offline_trajectory.duration, timestep):
        pos, vel, acc = offline_trajectory.at_time(t)
        offline_positions.append(pos[0])
        offline_velocities.append(vel[0])
        offline_accelerations.append(acc[0])
        
    return PositionVelocityAccelerationHistory(
        positions=offline_positions,
        velocities=offline_velocities,
        accelerations=offline_accelerations
    )

def test_waypoints_online():
    # Create instances: the Ruckig OTG as well as input and output parameters
    mock_motor = MockMotor()
    otg = Ruckig(1, mock_motor.mock_motor_config.time_step)  # DoFs, control cycle rate
    inp : InputParameter = InputParameter(1)  # DoFs
    out = OutputParameter(1, 10)  # DoFs, maximum number of intermediate waypoints for memory allocation
    
    # the data we read from moteus is the current position, velocity, and acceleration, not the target values
    inp.current_position = [mock_motor.previous.position]
    inp.current_velocity = [mock_motor.previous.velocity]
    inp.current_acceleration = [mock_motor.previous.acceleration]
    
    # Set target
    inp.target_position = [2] # meters
    inp.target_velocity = [0]
    inp.target_acceleration = [0]
    target_position_tolerance = 0.001 # meters - 1 mm
    target_velocity_tolerance = 0.005 # m/s - 5 mm/s

    # Set constraints
    inp.max_velocity = [1] # m/s
    inp.max_acceleration = [0.75 ] # m/s^2

    offline = generate_offline_trajectory(inp, mock_motor.mock_motor_config.time_step)

    # Generate the trajectory within the control loop
    out_list = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        out_list.append(copy(out))
        
        predicted_future = mock_motor.calculate_position_velocity_acceleration(out.new_velocity[0], out.new_acceleration[0])
        mock_motor.process_command(out.new_velocity[0], out.new_acceleration[0])
        
        # Use the predicted future position, velocity, and acceleration as the current values to account for moteus
        # read coming after write
        inp.current_position = [predicted_future.position]
        inp.current_velocity = [predicted_future.velocity]
        inp.current_acceleration = [predicted_future.acceleration]
        print(f"Motor Position: {mock_motor.current.position:.2f} Velocity: {mock_motor.current.velocity:.2f} Acceleration: {mock_motor.current.acceleration:.2f}")
        
        # Check if the current position is within the target position tolerance
        position_error = abs(mock_motor.current.position - inp.target_position[0])
        if position_error <= target_position_tolerance:
            print(f"Position error: {position_error} is within tolerance")
            velocity_error = abs(mock_motor.current.velocity - inp.target_velocity[0])
            if velocity_error <= target_velocity_tolerance:
                print(f"Velocity error: {velocity_error} is within tolerance")
                break  # Exit the loop if within tolerance

    # Plot the trajectory
    
    def plot_mock_motor_history(mock_motor, offline):
        fig = go.Figure()

        time_series = np.arange(0, len(mock_motor.history.positions) * mock_motor.mock_motor_config.time_step, mock_motor.mock_motor_config.time_step)
        
        fig.add_trace(go.Scatter(x=time_series, y=mock_motor.history.positions, mode='lines+markers', name='Position'))
        fig.add_trace(go.Scatter(x=time_series, y=mock_motor.history.velocities, mode='lines+markers', name='Velocity'))
        fig.add_trace(go.Scatter(x=time_series, y=mock_motor.history.accelerations, mode='lines+markers', name='Acceleration'))
        
        offline_time_series = np.arange(0, len(offline.positions) * mock_motor.mock_motor_config.time_step, mock_motor.mock_motor_config.time_step)
        
        fig.add_trace(go.Scatter(x=offline_time_series, y=offline.positions, mode='lines+markers', name='Offline Position'))
        fig.add_trace(go.Scatter(x=offline_time_series, y=offline.velocities, mode='lines+markers', name='Offline Velocity'))
        fig.add_trace(go.Scatter(x=offline_time_series, y=offline.accelerations, mode='lines+markers', name='Offline Acceleration'))

        fig.update_layout(
            title='Mock Motor History',
            xaxis_title='Time Step',
            yaxis_title='Value',
            legend_title='Legend'
        )

        fig.show()

    plot_mock_motor_history(mock_motor, offline)
    

