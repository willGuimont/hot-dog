var ghpages = require('gh-pages');

ghpages.publish(
    'public',
    {
        branch: 'gh-pages',
        repo: 'https://github.com/willGuimont/hot-dog-front',
        user: {
            name: 'William Guimont-Martin',
            email: 'william.guimont-martin.1@ulaval.ca'
        }
    },
    () => {
        console.log('Deploy Complete!')
    }
)