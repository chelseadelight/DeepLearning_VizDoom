import vizdoom as vzd
import numpy as np
import time

def main():
    game = vzd.DoomGame()
    game.load_config("scenarios/deathmatch.cfg")
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    game.new_episode()

    for _ in range(4):
        game.send_game_command("addbot")

    buttons = game.get_available_buttons()
    print("Buttons:", buttons)

    actions = []
    for i in range(len(buttons)):
        a = [0] * len(buttons)
        a[i] = 1
        actions.append(a)

    steps = 0
    while steps < 2000:
        if game.is_player_dead():
            game.respawn_player()
            game.advance_action(2)

        state = game.get_state()
        if state is None:
            game.advance_action()
            continue

        action = actions[np.random.randint(len(actions))]
        game.make_action(action, 4)

        time.sleep(0.01)
        steps += 1

    game.close()

if __name__ == "__main__":
    main()
