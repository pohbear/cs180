# CS180 Project Website

Created By Kenneth Poh Ren Kang, with the assistance of ChatGPT for HTML Template and design based on Bootstrap. All project work within is done by myself, AI was only used to beautify the website layout.

## Structure

```text
bootstrap-portfolio-template/
├── index.html
├── styles.css
├── assets/
│   └── images/
│       ├── part1-image-1.jpg
│       ├── part1-image-2.jpg
│       ├── part2-image-1.jpg
│       ├── part2-image-2.jpg
│       └── part3-demo.gif
└── projects/
    └── project-template.html
```

## Add a new project

1. Copy `projects/project-template.html`.
2. Rename it, e.g. `projects/my-project.html`.
3. Replace the titles, text, image filenames and captions.
4. Add the matching images/GIF to `assets/images/`.
5. Duplicate a project `<article>` in `index.html` and update its link.

## Add a new top-level page

1. Create a new HTML file in the repository root, e.g. `about.html`.
2. Copy the navbar from `index.html` into it.
3. Add `<li class="nav-item"><a class="nav-link" href="about.html">About</a></li>` to the navbar on each page.
4. On files inside `projects/`, use `../about.html` instead.

## Publish with GitHub Pages

1. Create a GitHub repository and upload/push these files.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose your main branch (usually `main`) and `/ (root)`.
5. Save. GitHub will publish the site after the Pages deployment completes.

If the repository is named `YOUR-USERNAME.github.io`, the site is served from the domain root. If it has another repository name, it is normally served under `/REPOSITORY-NAME/`. This template deliberately uses relative links so both cases work.
