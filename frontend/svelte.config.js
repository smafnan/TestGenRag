import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// BASE_PATH lets the Docker build mount the UI under /app (same origin as the
// API). Empty in local dev so everything is served from the root.
const base = process.env.BASE_PATH || '';

/** @type {import('@sveltejs/kit').Config} */
export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({ fallback: 'index.html', strict: false }),
    paths: { base },
  },
};
