import cProfile
import pstats
import train_tests


with cProfile.Profile() as pr:
    train_tests.main()


stats = pstats.Stats(pr)

stats.sort_stats(pstats.SortKey.TIME)

stats.dump_stats(filename="stats.prof")