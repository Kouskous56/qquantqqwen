"""Historical s30 recovery defaults; shared training implementation with s20."""
if __package__:
    from .recover_lora import main
else:
    from recover_lora import main


if __name__ == "__main__":
    main(default_scale="s30")
