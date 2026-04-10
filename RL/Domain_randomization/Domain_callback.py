from stable_baselines3.common.callbacks import BaseCallback
from Domain_randomization.randomization_gui import GUI
import threading
import time

try:
    from world_randomization_msgs.msg import Randomization
except ImportError:
    print("World randomization ROS messages not found, please ensure Domain Randomization AMBF Plugin is built and sourced")

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, env, randomization_args="0,0,1,1,1", seed=42, verbose=0):
        super().__init__(verbose)
        self.randomization_params = [True if x == "1" else False for x in str(randomization_args).split(",")]
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
        try:
            self.randomization_pub = self.ral_instance.publisher('/ambf/env/world_randomization/randomization', Randomization, queue_size=1)
            self.plugin_present = True
        except:
            self.plugin_present = False
        self._thread = None
        self._thread_stop = threading.Event()
        self.msg = Randomization()
        self.set_params()

    def _threaded_randomize(self):
        while not self._thread_stop.is_set():
            if self.plugin_present:
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
            if any(self.locals["dones"]) and self.plugin_present:
                self.set_params()           
        return True
    
    def set_params(self):
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

