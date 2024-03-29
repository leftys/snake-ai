#!/usr/bin/env python
import os
import sys
import traceback
import argparse
import contextlib
from importlib import import_module

try:
    from snakepit.robot_player import RobotPlayer, DEFAULT_SERVER_URL
    from snakepit.robot_snake import RobotSnake
except ImportError:
    print('snakepit Python package not found', file=sys.stderr)
    sys.exit(64)


ROBOT_CLASS_DEFAULT = 'snakepit.robot_snake.NoopRobotSnake'
ROBOT_FILE = None


def is_robot_class(value):
    return isinstance(value, type) and issubclass(value, RobotSnake) and value != RobotSnake


def robot_class(value):
    if '.' not in value:
        raise argparse.ArgumentTypeError('Invalid robot class path: "%s"' % value)

    module_name, class_name = value.rsplit('.', 1)

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        module_ = import_module(module_name)

    global ROBOT_FILE
    ROBOT_FILE = module_.__file__
    class_ = getattr(module_, class_name)

    if not is_robot_class(class_):
        raise argparse.ArgumentTypeError('Robot class "%s" does not inherit from RobotSnake' % class_)

    return class_


class RobotCode(argparse.FileType):
    """Return loaded json from file"""
    def __call__(self, filename):
        global ROBOT_FILE
        ROBOT_FILE = '<string>'
        fp = super(RobotCode, self).__call__(filename)
        code = compile(fp.read(), ROBOT_FILE, 'exec')

        if not code.co_names:
            raise argparse.ArgumentTypeError('The supplied code is empty')

        robot_ns = {}

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            exec(code, robot_ns)

        for key, val in robot_ns.items():
            if not key.startswith('_'):
                if is_robot_class(val):
                    class_ = val
                    break
        else:
            raise argparse.ArgumentTypeError('The code does not contain a RobotSnake-based class')

        globals().update(robot_ns)

        return class_


def validate_robot_class(robot_snake_class):
    from snakepit.world import World
    from snakepit.snake import Snake as ServerSnake

    world = World()  # Create test world
    robot_snake = robot_snake_class({}, world, 1)  # Create test player
    server_snake = ServerSnake({}, world, 1)  # Create dummy server snake

    for draw in server_snake.create():  # Simulate creating a new snake
        world.update(draw)  # Draw new snake to test world

    robot_snake.next_direction(initial=True)  # Call next_direction() for the first time
    del server_snake
    del robot_snake
    del world


def excepthook(etype, value, tb):
    if issubclass(etype, SyntaxError):
        print('{}: invalid syntax on line {}'.format(etype.__name__, value.lineno), file=sys.stderr)
    else:
        errors = []

        for err in traceback.extract_tb(tb):
            if err.filename == ROBOT_FILE:
                errors.append('  Error on line {} in {}\n    {}'.format(err.lineno, err.name, err.line))

        if errors:
            print('Traceback (most recent call last):', file=sys.stderr)
            print('\n'.join(errors), file=sys.stderr)

        print('{}: {}'.format(etype.__name__, value), file=sys.stderr)


parser = argparse.ArgumentParser(description='Run a robot snake; The code is loaded '
                                             'either from a module (--class) or from a string (--code).')
parser.add_argument('name', metavar='NAME', default='RobotSnake', nargs='?',
                    help='robot player name (default: RobotSnake)')
parser.add_argument('--id', dest='robot_id', default=None,
                    help='robot player ID (default: auto-generated by the server)')
group = parser.add_mutually_exclusive_group()
group.add_argument('--class', dest='class_', metavar='CLASS', type=robot_class,
                   help='robot snake class to be used (default: {})'.format(ROBOT_CLASS_DEFAULT))
group.add_argument('--code', dest='code_', metavar='FILE', type=RobotCode('r'),
                   help='robot snake code file (use "-" to read from stdin)')
parser.add_argument('--server', dest='server', metavar='URL', default=DEFAULT_SERVER_URL,
                    help='Snakepit server URL (default: {})'.format(DEFAULT_SERVER_URL))
parser.add_argument('--validate', dest='validate', action='store_true',
                    help='just validate the code and do not run it')

sys.excepthook = excepthook
args = parser.parse_args()
robot_name = args.name
robot_id = args.robot_id
server_url = args.server
robot_class = args.code_

if not robot_class:
    if args.class_:
        robot_class = args.class_
    else:
        robot_class = robot_class(ROBOT_CLASS_DEFAULT)
import main
robot_class = main.MyRobotSnake

if args.validate:
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        validate_robot_class(robot_class)
    sys.exit(0)

try:
    # Emojis on command line
    robot_name = robot_name.encode('utf8', 'surrogateescape').decode('utf8', 'surrogateescape')
except ValueError:
    pass

print('========  Creating new robot player "{!s}" using snake {!r} ======== '.format(robot_name, robot_class),
      file=sys.stderr)
sys.stderr.flush()
player = RobotPlayer(robot_name, player_id=robot_id, snake_class=robot_class, server_url=server_url)
player.run()
