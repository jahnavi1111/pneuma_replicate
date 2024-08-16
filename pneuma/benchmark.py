import cProfile
import io
import pstats
from pstats import SortKey

from pneuma import Pneuma


def profile_function(f, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    f(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(5)
    return s.getvalue()


def main():
    pneuma = Pneuma()
    print(profile_function(pneuma.purge_tables))


if __name__ == "__main__":
    main()
