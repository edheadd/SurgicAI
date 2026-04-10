import time
from std_msgs.msg import Empty
from ambf_msgs.msg import WorldCmd

class WorldManager:
    """Manages world state and reset commands"""
    
    def __init__(self, ral_instance):
        """
        Initialize WorldManager with RAL instance.
        
        :param ral_instance: RAL instance for ROS communication
        """
        self.ral = ral_instance
        self._world_cmd = WorldCmd()
        self._reset_pub = self.ral.publisher(
            '/ambf/env/World/Command/Reset',
            Empty,
            queue_size=1
        )
        self._world_cmd_pub = self.ral.publisher(
            '/ambf/env/World/Command',
            WorldCmd,
            queue_size=1
        )
        time.sleep(0.5)  # Allow some time for publishers to initialize
    
    def reset(self):
        """Reset the world state"""
        reset_msg = Empty()
        self._reset_pub.publish(reset_msg)
        #print("Published world reset command")
        time.sleep(0.5)
        
    def update(self):
        self._world_cmd.step_clock = not self._world_cmd.step_clock
        self._world_cmd_pub.publish(self._world_cmd)
