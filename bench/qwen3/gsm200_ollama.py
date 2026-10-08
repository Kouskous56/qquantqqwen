"""GSM8K gsm200 runner. Historical sources are retained in frozen/."""
if __package__:
    from .common import extract, gsm_main
else:
    from common import extract, gsm_main


def main(argv=None):
    return gsm_main("gsm200", __file__, argv)


if __name__ == "__main__":
    main()
