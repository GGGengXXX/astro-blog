---
title: 'Astro 博客维护清单'
description: '给未来自己的维护备忘：新增文章、改首页、部署和排错。'
pubDate: 'Jun 20 2026'
heroImage: '../../assets/blog-placeholder-4.jpg'
category: 'note'
mood: '清醒'
tags: ['Astro', 'GitHub Pages', '笔记']
featured: true
---

这份清单留给未来的自己。

## 写新文章

在 `src/content/blog/` 里新建一个 `.md` 或 `.mdx` 文件，文件名会变成 URL。

Frontmatter 可以这样写：

```yaml
---
title: '文章标题'
description: '一句话摘要'
pubDate: 'Jun 21 2026'
category: 'note'
mood: '专注'
tags: ['Astro', '笔记']
---
```

## 本地预览

```sh
npm install
npm run dev
```

## 发布

推送到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages。

如果页面路径不对，优先检查 `astro.config.mjs` 里的 `site` 和 `base`。
