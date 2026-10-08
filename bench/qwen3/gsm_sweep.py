"""GSM8K sweep runner. Historical sources are retained in frozen/."""
if __package__:
    from .common import extract, gsm_main
else:
    from common import extract, gsm_main


def main(argv=None):
    return gsm_main("sweep", __file__, argv)


if __name__ == "__main__":
    main()
