from ros_abstraction_layer import ral
import time
from utils.psm_arm import PSM

def main():
    ral_instance = ral("ral_test_node")
    print("Initializing PSM...")
    psm = PSM(ral_instance, name="psm2")
    time.sleep(1)  # Allow some time for connections to establish
    ral_instance.spin()

    print("Sending measured pose as command at 20 Hz. Press Ctrl+C to exit.")
    try:
        input("Press ENTER to start sending commands...")
        while True:
            measured_pose = psm.measured_cp()
            if measured_pose is not None:
                print(measured_pose)
                #psm.servo_cp(measured_pose)
                #print("Sent measured pose as command.")
            else:
                print("Measured pose not available yet.")
            time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("Shutting down test.")
        ral_instance.shutdown()

if __name__ == "__main__":
    main()