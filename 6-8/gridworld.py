class GridWorld:
    def __init__(self, grid, move_value = -1, die_value = -20, win_value = 10):
        self.grid = self.parse(grid)
        self.move_value = move_value
        self.die_value = die_value
        self.win_value = win_value

    def move_dir(self, from_state, dir):
        to_state = {
            0 : (from_state[0], from_state[1] - 1),
            1 : (from_state[0] + 1, from_state[1]),
            2 : (from_state[0], from_state[1] + 1),
            3 : (from_state[0] - 1, from_state[1]),
        }.get(dir)
        return self.move(from_state, to_state)

    def get_state_reward(self, state):
        x,y = state
        field = self.grid[x][y]
        return self.get_field_reward(field)

    def get_field_reward(self, field):
        value = {
            'P' : self.die_value,
            'G' : self.win_value,
        }.get(field, self.move_value)
        return value

    def move(self, from_state, to_state):
        width, height = len(self.grid), len(self.grid[0])
        x, y = to_state

        out_of_bound = x < 0 or x > width - 1 or y < 0 or y > height -1
        if(out_of_bound):
            to_field = ' '
        else:
            to_field = self.grid[x][y]

        new_state = to_state
        if(out_of_bound or to_field == '#'):
            new_state = from_state

        is_terminal = to_field == 'P' or to_field == 'G'

        value = self.get_field_reward(to_field)

        return new_state, value, is_terminal
    
    def parse(self, grid_string):
        lines = grid_string.split('\n')
        width, height = len(lines[0]), len(lines)

        grid = [[' ' for y in range(height)] for x in range(width)]
        for x in range(width):
            for y in range(height):
                grid[x][y] = lines[y][x]
        return grid

    def print(self):
        width, height = len(self.grid), len(self.grid[0])

        lines = []
        for y in range(height):
            lines.append(''.join([self.grid[x][y] for x in range(width)]))

        print('\n'.join(lines))