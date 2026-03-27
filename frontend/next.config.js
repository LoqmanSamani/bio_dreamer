/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/bio_dreamer",
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

module.exports = nextConfig;
