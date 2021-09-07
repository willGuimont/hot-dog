npm run build
git add .
git commit -m "deploy"
git push -f origin gh-pages
node ./gh-pages.js
