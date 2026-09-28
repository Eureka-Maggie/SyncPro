# Contributing

Thank you for helping improve SyncPro. Focused bug fixes, documentation corrections, portability improvements, and reproducibility reports are welcome.

Before opening a pull request:

1. Create a branch from `main` and keep the change scoped.
2. Do not commit videos, frames, model checkpoints, credentials, local paths, or benchmark outputs.
3. Run `python -m unittest tests.test_release_contract`.
4. Run `git diff --check` and describe the validation performed.
5. Preserve third-party copyright and license notices.

For data issues, include the configuration, split, `sample_id` or `video_id`, and a description of the annotation problem. Do not attach restricted source media to a public issue.
