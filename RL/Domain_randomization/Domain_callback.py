from stable_baselines3.common.callbacks import BaseCallback
from Domain_randomization.randomization_gui import GUI
import threading
import time
from typing import Optional

try:
    from world_randomization_msgs.msg import Randomization
except ImportError:
    print("World randomization ROS messages not found, please ensure Domain Randomization AMBF Plugin is built and sourced")
    Randomization = None

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, env, randomization_args="0,0,1,1,1", seed=42, verbose=0):
        super().__init__(verbose)
        """
        Stable-Baselines3 callback that periodically publishes a ROS Randomization message.

        This is intended to drive AMBF's domain-randomization plugin while training.
        When the plugin (or message type) is not available, the callback becomes a no-op.
        """
        self.randomization_params = [True if x == "1" else False for x in randomization_args.split(",")]
        print(f"Domain Randomization parameters: {self.randomization_params}")
        self.env = env
        self.ral_instance = self.env.unwrapped.ral_instance
        self.seed = seed
        self.name_dict = {
            "gravity": "Gravity vector in the world frame.",
            "friction": "Friction of the objects in the scene.",
            "light_num": "Number of lights in the scene.",
            "light_color": "Color of the lights in the scene.",
            "light_attenuation": "Attenuation of the lights in the scene.",
            "shadows": "Shadow presence in the scene.",
            "shaders": "Shaders used in the scene."
        }
        # The plugin is considered "present" only if we can import the message type and
        # create a publisher. This keeps training usable on machines without ROS msgs.
        self.plugin_present = False
        self.randomization_pub = None
        if Randomization is not None:
            try:
                self.randomization_pub = self.ral_instance.publisher(
                    '/ambf/env/world_randomization/randomization',
                    Randomization,
                    queue_size=1
                )
                self.plugin_present = True
            except Exception:
                self.plugin_present = False
        self._thread = None
        self._thread_stop = threading.Event()
        self.msg: Optional[object] = Randomization() if (self.plugin_present and Randomization is not None) else None
        if self.msg is not None:
            self.set_params()

    def _threaded_randomize(self):
        while not self._thread_stop.is_set():
            if self.plugin_present and self.randomization_pub is not None and self.msg is not None:
                self.msg.timestep = self.num_timesteps
                self.randomization_pub.publish(self.msg)
            time.sleep(1)

    def _on_rollout_start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread_stop.clear()
            self._thread = threading.Thread(target=self._threaded_randomize, daemon=True)
            self._thread.start()
            
    def _on_training_end(self):
        # Stop the thread when training ends
        if self._thread is not None:
            self._thread_stop.set()
            self._thread.join()
            self._thread = None

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]) and self.plugin_present and self.msg is not None:
                self.set_params()           
        return True
    
    def set_params(self):
        if self.msg is None:
            return
        self.msg.seed = self.seed
        self.msg.gravity = self.randomization_params[0]
        self.msg.friction = self.randomization_params[1]
        self.msg.light_num = self.randomization_params[2]
        self.msg.light_color = self.randomization_params[3]
        self.msg.light_attenuation = self.randomization_params[4]
        # self.msg.shadows = self.randomization_params[5]
        # self.msg.shader = self.randomization_params[6]
        
    def start_gui(self, app):
        if self.plugin_present:
            self.gui_window = GUI(self)
            self.gui_window.show()
        
    def update_randomization_params(self, idx, immediate_randomization):
        self.randomization_params[idx] = not self.randomization_params[idx]
        if immediate_randomization:
            self.set_params()
    
    def reset_env(self):
        self.env.unwrapped.reset()
        self.set_params()
        
        
    def start_thread(self):
        """Manually start the randomization thread."""
        if self._thread is None or not self._thread.is_alive():
            self._thread_stop.clear()
            self._thread = threading.Thread(target=self._threaded_randomize, daemon=True)
            self._thread.start()
