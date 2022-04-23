"""
Define SSBM types. 
"""

from ctypes import *
from .ctype_util import *
from enum import IntEnum
import struct
import tempfile
import os
#import h5py
import pickle
import math
import itertools
import attr
from .state import Character

@pretty_struct
class Stick(Structure):
  _fields = [
    ('x', c_float),
    ('y', c_float),
  ]

  def __init__(self, x=0.5, y=0.5):
    self.x = x
    self.y = y

  def reset(self):
    self.x = 0.5
    self.y = 0.5

  def __sub__(self, other):
    return 2 * (self.x - other.x), 2 * (self.y - other.y)
  
  @classmethod
  def polar(cls, theta, r=1.):
    r /= 2.
    return cls(x=0.5+r*math.cos(theta), y=0.5+r*math.sin(theta))

@pretty_struct
class RealControllerState(Structure):
  _fields = [
    ('button_A', c_bool),
    ('button_B', c_bool),
    ('button_X', c_bool),
    ('button_Y', c_bool),
    ('button_Z', c_bool),
    ('button_L', c_bool),
    ('button_R', c_bool),
    ('button_START', c_bool),

    ('analog_L', c_float),
    ('analog_R', c_float),

    ('stick_MAIN', Stick),
    ('stick_C', Stick),
  ]

  def __init__(self):
    self.reset()

  def reset(self):
    "Resets controller to neutral position."
    self.button_A = False
    self.button_B = False
    self.button_X = False
    self.button_Y = False
    self.button_L = False
    self.button_R = False

    self.analog_L = 0.0
    self.analog_R = 0.0

    self.stick_MAIN.reset()
    self.stick_C.reset()

  def __sub__(self, other):
    return (self.button_A - other.button_A,
    self.button_B - other.button_B,
    self.button_X - other.button_X,
    self.button_Y - other.button_Y,
    self.button_L - other.button_L,
    self.button_R - other.button_R,
    self.analog_L - other.analog_L,
    self.analog_R - other.analog_R,
    *(self.stick_MAIN - other.stick_MAIN), 
    *(self.stick_C - other.stick_C))
  
RealControllerState.neutral = RealControllerState()

@pretty_struct
class PlayerMemory(Structure):
  _fields = [
    ('percent', c_uint),
    ('stock', c_uint),
    # 1.0 is right, -1.0 is left
    ('facing', c_float),
    ('x', c_float),
    ('y', c_float),
    ('z', c_float),
    ('action_state', c_uint),
    ('action_counter', c_uint),
    ('action_frame', c_float),
    ('character', c_uint),
    ('invulnerable', c_bool),
    ('hitlag_frames_left', c_float),
    ('hitstun_frames_left', c_float),
    ('jumps_used', c_uint),
    ('charging_smash', c_bool),
    ('in_air', c_bool),
    ('speed_air_x_self', c_float),
    ('speed_ground_x_self', c_float),
    ('speed_y_self', c_float),
    ('speed_x_attack', c_float),
    ('speed_y_attack', c_float),
    ('shield_size', c_float),
    ('lightshield_amount', c_float),

    ('cursor_x', c_float),
    ('cursor_y', c_float),

    # NOTE: the sticks here are [-1, 1],
    # not [0, 1] like in pad.py
    ('controller', RealControllerState)
  ]

@pretty_struct
class GameMemory(Structure):
  _fields = [
    ('players', PlayerMemory * 2),

    ('frame', c_uint),
    ('menu', c_uint),
    ('stage', c_uint),
    
    # stage select screen
    ('sss_cursor_x', c_float),
    ('sss_cursor_y', c_float),
  ]

class SimpleButton(IntEnum):
  NONE = 0
  A = 1
  B = 2
  Z = 3
  Y = 4
  L = 5

neutral_stick = (0.5, 0.5)

@attr.s
class SimpleController(object):
  button = attr.ib(default=SimpleButton.NONE)
  stick = attr.ib(default=neutral_stick)
  duration = attr.ib(default=None)
  
  @classmethod
  def init(cls, *args, **kwargs):
    self = cls(*args, **kwargs)
    self.real_controller = self.realController()
    return self
  
  def realController(self):
    controller = RealControllerState()
    if self.button is not SimpleButton.NONE:
      setattr(controller, "button_%s" % self.button.name, True)

    controller.stick_MAIN = self.stick
    return controller
  
  def banned(self, player, char):
    # toad does weird things. needs investigation
    if char == 'peach' and self.button == SimpleButton.B and self.stick == neutral_stick:
      return True
    # transforming screws up the memory watcher
    if char in ['sheik', 'zelda'] and  self.button == SimpleButton.B and self.stick[1] < 0.5:
      return True
    # fox dittos can end up in infinite lasers
    #if char == 'fox' and self.button == SimpleButton.B and self.stick == neutral_stick:
    #  return True
    # fox and falco suck at recovering because they learn to just shine-stall
    # they also learn to never side-b because it sometimes leads to instant death
    if char in ['fox', 'falco'] and self.button == SimpleButton.B:
      if self.stick[0] > 0.5 and player.x > 0: return True  # don't side-B off the right
      if self.stick[0] < 0.5 and player.x < 0: return True  # don't side-B off the left
      if self.stick[1] < 0.5:  # shine stall 
        if abs(player.x) > 100 or player.y < -5: return True

    if char == 'puff':
      # side-B spam to avoid dying is lame
      if (player.jumps_used >= 6 and
          self.button == SimpleButton.B and self.stick[0] != 0.5 and
          player.y < -5):
        return True

    return False
  
  def send(self, pad, player, char):
    if self.banned(player, char):
      pad.send_controller(RealControllerState.neutral)
    else:
      pad.send_controller(self.real_controller)

SimpleController.neutral = SimpleController.init()


class RepeatController(object):
  duration = None

  def send(self, pad, *args):
    pass

repeat_controller = RepeatController()

axis_granularity = 3
axis_positions = [i/(axis_granularity-1) for i in range(axis_granularity)]#np.linspace(0, 1, axis_granularity)
diagonal_sticks = list(itertools.product(axis_positions, repeat=2))
diagonal_controllers = [SimpleController.init(*args) for args in itertools.product(SimpleButton, diagonal_sticks)]


class ActionChain(object):
  """
  A list of actions, each with a duration, and the last duration must be None.
  
  TODO: Come up with a better system?
  """

  def __init__(self, action_list, act_every):
    self.actions = []
    for action in action_list:
      if action.duration:
        self.actions += [action] * action.duration
      else:
        self.actions += [action] * (act_every - len(self.actions))
    assert len(self.actions) == act_every
    self.index = 0

  def act(self, pad, *args):
    self.actions[self.index].send(pad, *args)
    self.index += 1
  
  def done(self):
    return self.index == len(self.actions)


class ActionSet(object):
  def __init__(self, actions):
    self.actions = list(map(lambda obj: obj if isinstance(obj, list) else [obj], actions))
    self.size = len(actions)
  
  def choose(self, index, act_every):
    return ActionChain(self.actions[index], act_every)

old_sticks = [(0.5, 0.5), (0.5, 1), (0.5, 0), (0, 0.5), (1, 0.5)]
old_controllers = [SimpleController.init(*args) for args in itertools.product(SimpleButton, old_sticks)]

cardinal_sticks = [(0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1), (0.5, 0.5)]
cardinal_controllers = [SimpleController.init(*args) for args in itertools.product(SimpleButton, cardinal_sticks)]

tilt_sticks = [(0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6)]

custom_controllers = itertools.chain(
  itertools.product([SimpleButton.A, SimpleButton.B], cardinal_sticks), #smashes and specials
  itertools.product([SimpleButton.A], tilt_sticks), #tilts
  itertools.product([SimpleButton.NONE, SimpleButton.L], diagonal_sticks), #shield
  itertools.product([SimpleButton.Z, SimpleButton.Y], [neutral_stick]), #zair
)
custom_controllers = [SimpleController.init(*args) for args in custom_controllers]
custom_controllers.append(repeat_controller)

grab = SimpleController.init(button=SimpleButton.Z)
jb = SimpleController.init(button=SimpleButton.A)

# allows fox, sheik, samus, etc to short hop with act_every=3
short_hop = SimpleController.init(button=SimpleButton.Y, duration=2)
short_hop_chain = [short_hop, SimpleController.neutral]

full_hop = SimpleController.init(button=SimpleButton.Y, duration=4)

# better sh that also allows jc grab and upsmash at act_every 3
sh2_chain = [
  SimpleController.init(duration=2),
  SimpleController.init(button=SimpleButton.Y),
]

# wd_left_long = SimpleController.init(button=SimpleButton.L, stick=Stick.polar(-7/8 * math.pi))
# wd_right_long = SimpleController.init(button=SimpleButton.L, stick=Stick.polar(-1/8 * math.pi))
# wd_left_short = SimpleController.init(button=SimpleButton.L, stick=Stick.polar(-7/8 * math.pi))
# wd_right_short = SimpleController.init(button=SimpleButton.L, stick=Stick.polar(-1/8 * math.pi))
#general purpose wavedashing
wd_all = [[
  SimpleController.init(button=SimpleButton.Y, duration=3),
  SimpleController.init(button=SimpleButton.L, stick=Stick.polar(-i/16 * math.pi))
] for i in range(1,16) if i != 4] 

#full hop aerials

actionTypes = dict(
  old = ActionSet(old_controllers),
  cardinal = ActionSet(cardinal_controllers),
  diagonal = ActionSet(diagonal_controllers),
  custom = ActionSet(custom_controllers),
  # short_hop_test = ActionSet([SimpleController.neutral] * 10 + [short_hop_chain]),
  # short_hop = ActionSet(custom_controllers + [short_hop]),
  # custom_sh_jc = ActionSet(custom_controllers + [short_hop_chain, jc_chain]),
  # fox_wd_test = ActionSet([SimpleController.neutral] * 10 + [fox_wd_chain_left]),
  custom_sh2_wd = ActionSet(custom_controllers + [sh2_chain] + wd_all),
  wd_attack = ActionSet(wd_all, custom_controllers),
  wd_grab = ActionSet(wd_all + [grab]),
  sh_aerials = ActionSet([short_hop] + custom_controllers), #short hop aerials
  sh_aerials = ActionSet([full_hop] + custom_controllers), #full hop aerials
  throws = ActionSet([grab] + cardinal_controllers),
  throws_pummel_1 = ActionSet([grab] + [jb] + cardinal_controllers),
  throws_pummel_2 = ActionSet([grab] + [jb] * 2 + cardinal_controllers),
  throws_pummel_3 = ActionSet([grab] + [jb] * 3 + cardinal_controllers),
)

@pretty_struct
class SimpleStateAction(Structure):
  _fields = [
    ('state', GameMemory),
    ('prev_action', c_uint),
    ('action', c_uint),
    ('prob', c_float),
  ]

