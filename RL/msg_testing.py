#!/usr/bin/env python3
"""
One-shot ROS publisher for world_randomization_msgs/Randomization.
Edit the values in the `randomization_params` and `other_fields` at the top, then run this script.
It waits briefly for subscribers, publishes once, sleeps to allow the message to be sent, then exits.
"""

import time
import rospy

try:
    from world_randomization_msgs.msg import Randomization
except ImportError:
    raise SystemExit("world_randomization_msgs.msg.Randomization not found. Make sure your ROS environment is sourced and the message package is available.")

# --- Edit these values before running -------------------------------------------------
# Format: [gravity, friction, light_num, light_color, light_attenuation, shadows, shader]
randomization_params = [False, False, False, False, False, False, True]
other_fields = {
    "timestep": 0,
    "seed": 41,
}
# --------------------------------------------------------------------------------------


def main():
    rospy.init_node('msg_testing', anonymous=True, disable_signals=True)
    topic = '/ambf/env/world_randomization/randomization'
    pub = rospy.Publisher(topic, Randomization, queue_size=1)

    # Wait for a subscriber (short timeout)
    timeout = 5.0
    start = time.time()
    while pub.get_num_connections() == 0:
        if time.time() - start > timeout:
            print(f"Warning: no subscribers connected after {timeout} seconds. Publishing anyway to {topic}.")
            break
        time.sleep(0.05)

    msg = Randomization()
    msg.timestep = other_fields.get('timestep', 0)
    msg.seed = other_fields.get('seed', 42)

    # Assign the fields from the list (guarded by attribute existence)
    try:
        msg.gravity = randomization_params[0]
        msg.friction = randomization_params[1]
        msg.light_num = randomization_params[2]
        msg.light_color = randomization_params[3]
        msg.light_attenuation = randomization_params[4]
        msg.shadows = randomization_params[5]
        msg.shader = randomization_params[6]
    except Exception as e:
        print('Error assigning randomization fields:', e)
        raise

    pub.publish(msg)
    print(f"Published Randomization to {topic}:\n  timestep={msg.timestep} seed={getattr(msg, 'seed', None)}\n  params={randomization_params}")

    # Give ROS time to send the message
    time.sleep(0.5)


if __name__ == '__main__':
    main()
