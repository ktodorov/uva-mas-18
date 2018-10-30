import sys
import unittest
sys.path.insert(0,'../')
from gridworld import GridWorld

class GridWorldTest(unittest.TestCase):

    def test_parse(self):
        grid = ' #P\nG #'
        gw = GridWorld(grid)
        self.assertEqual(gw.grid[1][0], '#')
        self.assertEqual(gw.grid[2][0], 'P')
        self.assertEqual(gw.grid[0][1], 'G')
        self.assertEqual(gw.grid[1][1], ' ')

    def test_move(self):
        grid = ' #P\nG #'
        gw = GridWorld(grid, move_value=-1, die_value=-20, win_value=10)

        step_tests = [
            # move into wall
            ((0,0), (1,0), (0,0), -1, False),
            # move to free field
            ((0,0), (1,1), (1,1), -1, False),
            # move to goal
            ((0,0), (0,1), (0,1), 10, True),
            # die penalty
            ((0,0), (2,0), (2,0), -20, True),
            # out of bounds #1
            ((0,0), (-1,0), (0,0), -1, False),
            # out of bounds #1
            ((0,0), (10,0), (0,0), -1, False),
        ]

        for start, to, end, reward, is_terminal in step_tests:
            e, r, t = gw.move(start, to)
            self.assertEqual(e, end)
            self.assertEqual(r, reward)
            self.assertEqual(t, is_terminal)

    def test_move_dir(self):
        grid = '   \n   \n   '
        gw = GridWorld(grid)

        start = (1,1)
        # N, E, S, W
        tests = [(0, (1,0)), (1, (2,1)), (2, (1,2)), (3, (0,1))]

        for dir, end in tests:
            e, _, _ = gw.move_dir(start, dir)
            self.assertEqual(e, end)


if __name__ == '__main__':
    unittest.main()