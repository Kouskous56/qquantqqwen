"""GSM8K harness runner. Historical sources are retained in frozen/."""
if __package__:
    from .common import extract, gsm_main
else:
    from common import extract, gsm_main


def main(argv=None):
    return gsm_main("harness", __file__, argv)


if __name__ == "__main__":
    main()
