# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motor model for quadrupeds."""

import collections
import numpy as np

# from robots import robot_config
# class MotorControlMode(enum.Enum):
#   """The supported motor control modes."""
#   POSITION = 1,

#   # Apply motor torques directly.
#   TORQUE = 2,

#   # Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
#   # position and velocities. kp and kd are PD gains. tau is the additional
#   # motor torque. This is the most flexible control mode.
#   HYBRID = 3,

#   # PWM mode is only availalbe for Minitaur
#   PWM = 4

NUM_MOTORS = 12

MOTOR_COMMAND_DIMENSION = 5

# These values represent the indices of each field in the motor command tuple
POSITION_INDEX = 0
POSITION_GAIN_INDEX = 1
VELOCITY_INDEX = 2
VELOCITY_GAIN_INDEX = 3
TORQUE_INDEX = 4

CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD", "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE" ]

class QuadrupedMotorModel(object):
  """A simple motor model for Laikago.

    When in POSITION mode, the torque is calculated according to the difference
    between current and desired joint angle, as well as the joint velocity.
    For more information about PD control, please refer to:
    https://en.wikipedia.org/wiki/PID_controller.

    The model supports a HYBRID mode in which each motor command can be a tuple
    (desired_motor_angle, position_gain, desired_motor_velocity, velocity_gain,
    torque).

  """

  def __init__(self,
               kp=60,
               kd=1,
               torque_limits=None,
               motor_control_mode="PD"):
    self._kp = kp
    self._kd = kd
    self._torque_limits = torque_limits
    if torque_limits is not None:
      if isinstance(torque_limits, (collections.Sequence, np.ndarray)):
        self._torque_limits = np.asarray(torque_limits)
      else:
        self._torque_limits = np.full(NUM_MOTORS, torque_limits)
    self._motor_control_mode = motor_control_mode
    self._strength_ratios = np.full(NUM_MOTORS, 1)

  def set_strength_ratios(self, ratios):
    """Set the strength of each motors relative to the default value.

    Args:
      ratios: The relative strength of motor output. A numpy array ranging from
        0.0 to 1.0.
    """
    self._strength_ratios = ratios

  def set_motor_gains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
    self._kp = kp
    self._kd = kd

  def set_voltage(self, voltage):
    pass

  def get_voltage(self):
    return 0.0

  def set_viscous_damping(self, viscous_damping):
    pass

  def get_viscous_dampling(self):
    return 0.0

  def convert_to_torque(self,
                        motor_commands,
                        motor_angle,
                        motor_velocity,
                        motor_control_mode=None):
    """Convert the commands (position control or torque control) to torque.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      motor_angle: The motor angle observed at the current time step. It is
        actually the true motor angle observed a few milliseconds ago (pd
        latency).
      motor_velocity: The motor velocity observed at the current time step, it
        is actually the true motor velocity a few milliseconds ago (pd latency).
      true_motor_velocity: The true motor velocity. The true velocity is used to
        compute back EMF voltage and viscous damping.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    #del true_motor_velocity
    if not motor_control_mode:
      motor_control_mode = self._motor_control_mode

    # if motor_control_mode is robot_config.MotorControlMode.PWM:
    #   raise ValueError(
    #       "{} is not a supported motor control mode".format(motor_control_mode))

    # No processing for motor torques
    # Edit: SHOULD still clip torque values
    if motor_control_mode is "TORQUE":
      assert len(motor_commands) == NUM_MOTORS
      motor_torques = self._strength_ratios * motor_commands
      motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                              self._torque_limits)
      #print('actual motor', motor_torques)
      return motor_torques, motor_torques

    desired_motor_angles = None
    desired_motor_velocities = None
    kp = None
    kd = None
    additional_torques = np.full(NUM_MOTORS, 0)
    if motor_control_mode is "PD":
      assert len(motor_commands) == NUM_MOTORS
      kp = self._kp
      kd = self._kd
      desired_motor_angles = motor_commands
      desired_motor_velocities = np.full(NUM_MOTORS, 0)
    else:
      raise ValueError("Motor model should only be torque or position control.")
    # elif motor_control_mode is robot_config.MotorControlMode.HYBRID:
    #   # The input should be a 60 dimension vector
    #   assert len(motor_commands) == MOTOR_COMMAND_DIMENSION * NUM_MOTORS
    #   kp = motor_commands[POSITION_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
    #   kd = motor_commands[VELOCITY_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
    #   desired_motor_angles = motor_commands[
    #       POSITION_INDEX::MOTOR_COMMAND_DIMENSION]
    #   desired_motor_velocities = motor_commands[
    #       VELOCITY_INDEX::MOTOR_COMMAND_DIMENSION]
    #   additional_torques = motor_commands[TORQUE_INDEX::MOTOR_COMMAND_DIMENSION]
    motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (
        motor_velocity - desired_motor_velocities) + additional_torques
    motor_torques = self._strength_ratios * motor_torques
    if self._torque_limits is not None:
      if len(self._torque_limits) != len(motor_torques):
        raise ValueError(
            "Torque limits dimension does not match the number of motors.")
      motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                              self._torque_limits)

    return motor_torques, motor_torques
