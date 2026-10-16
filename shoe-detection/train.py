"""Compatibility entry point: training only; evaluate test separately."""
from pipeline.run import main

if __name__ == '__main__':
    main(action='train')