# Blog Posts

This directory contains all your blog posts. Each post is a standalone HTML file.

## How to Add a New Blog Post

### Step 1: Create the Post File

1. Copy the template file:
   ```bash
   cp posts/blog-post-template.html posts/your-post-name.html
   ```

2. Edit the new file and update:
   - `<title>` tag with your post title
   - Date in the `post-meta` section
   - Category/tag in the `post-meta` section
   - `<h1>` with your post title
   - The main content

### Step 2: Add the Post to blog.html

Open `blog.html` and add a new article block in the `blog-posts` section:

```html
<article class="blog-post">
    <div class="post-meta">
        <span class="date">Your Date</span>
        <span class="tag">Your Category</span>
    </div>
    <h2><a href="posts/your-post-name.html">Your Post Title</a></h2>
    <p class="excerpt">
        A brief excerpt or summary of your post (1-2 sentences).
    </p>
    <a href="posts/your-post-name.html" class="read-more">Read more →</a>
</article>
```

Add this **above** the empty-state div, so the newest posts appear first.

### Step 3: Commit and Push

```bash
git add posts/your-post-name.html blog.html
git commit -m "Add new blog post: Your Post Title"
git push origin main
```

Then update the gh-pages branch following the same process as before.

## Formatting Tips

### Code Blocks

Use `<pre><code>` tags for code:

```html
<pre><code>def example():
    return "Hello, World!"</code></pre>
```

### Images

Place images in an `images/` directory and reference them:

```html
<img src="../images/your-image.png" alt="Description">
```

### Math Equations

To add LaTeX math support, include MathJax in your post's `<head>`:

```html
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

Then use `$$` for display math or `\(` for inline math.

### Syntax Highlighting

For syntax highlighting, you can add [Prism.js](https://prismjs.com/) or [highlight.js](https://highlightjs.org/).

## Alternative: Use a Static Site Generator

For easier blog management, consider:
- **Jekyll**: Integrates directly with GitHub Pages
- **Hugo**: Very fast static site generator
- **nbdev/Quarto**: Write posts in Jupyter notebooks

These tools let you write in Markdown instead of HTML.
