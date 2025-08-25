import numpy as np
import pygame
from Approach_env import SRC_approach
from Insert_env import SRC_insert
from Regrasp_env import SRC_regrasp
from Pullout_env import SRC_pullout
import time

KEY2ACTION = {
    pygame.K_w: (0, -0.5),
    pygame.K_a: (1, -0.5),
    pygame.K_d: (2,  0.5),
    pygame.K_UP: (0, 0.5),
    pygame.K_LEFT: (1, 0.5),
    pygame.K_RIGHT: (2, -0.5),
    pygame.K_DOWN: (6, -0.5),
}

def teleop_loop():
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Teleop Control")

    trans_step = 1.0e-3
    angle_step = np.deg2rad(3)
    jaw_step = 0.05
    step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)

    threshold = [3,np.deg2rad(30)]

    env = SRC_approach(step_size=step_size, threshold=threshold)
    #env = SRC_insert(step_size=step_size, threshold=threshold)
    #env = SRC_regrasp(step_size=step_size, threshold=threshold)
    #env = SRC_pullout(step_size=step_size, threshold=threshold)
    env.reset()

    running = True
    clock = pygame.time.Clock()

    while running:
        action = np.zeros(7, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        # for key, (idx, val) in KEY2ACTION.items():
        #     if keys[key]:
        #         action[idx] = val

        

        if keys[pygame.K_ESCAPE]:
            running = False

        obs, reward, done, truncate, info = env.step(action)
        if done:
            print("Goal reached")
            time.sleep(20)
            env.reset()

        clock.tick(50)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    teleop_loop()
