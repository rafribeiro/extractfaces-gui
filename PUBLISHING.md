# Publishing releases

The workflow `.github/workflows/publish.yml` tests and builds the app on
Windows, then publishes it to PyPI when a GitHub release is published.
It uses Trusted Publishing, so no PyPI API token is stored in the repository.

## One-time setup

1. Upload the project to the public GitHub repository
   `rafribeiro/extractfaces-gui`.
2. Create a GitHub environment named `pypi` in the repository's
   **Settings → Environments**.
3. Sign into PyPI and open https://pypi.org/manage/account/publishing/.
   For this new package, add a pending GitHub publisher with:
   - PyPI project name: `extractfaces-gui`
   - Owner: `rafribeiro`
   - Repository: `extractfaces-gui`
   - Workflow filename: `publish.yml`
   - Environment: `pypi`

The PyPI name must be available; a pending publisher does not reserve it.
If the project already exists under your account, add the trusted publisher
from that project's **Publishing** settings instead.

## Release version 0.1.0

1. Commit and push the code, including the workflow.
2. On GitHub, open **Releases → Draft a new release**.
3. Create tag `v0.1.0` targeting the commit you want to publish.
4. Publish the release. This starts the workflow and uploads to PyPI.
5. Check the repository's **Actions** tab for the result.

After success, users can install with `pip install extractfaces-gui` and launch
with `extract-faces`. They need Python 3.14 or newer and the detector model
described in README.md.

For later releases, increase the version in `pyproject.toml`, refresh the
lockfile with `uv lock`, commit the changes, and create a matching `vVERSION`
release. PyPI does not allow replacing files from an already uploaded release.

Reference: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## Upload from Windows

Install Git and GitHub CLI, then open a new terminal and run:

```powershell
gh auth login --hostname github.com --git-protocol https --web
git init -b main
git config user.name "Rafael O. Ribeiro"
git config user.email "rafaeloliveiraribeiro@gmail.com"
git add .gitignore .python-version README.md PUBLISHING.md LICENSE pyproject.toml uv.lock src tests .github
git commit -m "Initial release of Extract Faces GUI"
gh repo create rafribeiro/extractfaces-gui --public --source . --remote origin --push
```

These commands assume a new repository, with no existing Git history or remote.
Configure the PyPI pending publisher before publishing the first GitHub release.
