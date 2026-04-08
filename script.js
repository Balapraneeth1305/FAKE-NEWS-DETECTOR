function detectFromLink() {
    const link = document.getElementById('link-input').value;
    if (link) {
        detectText(link);
    } else {
        document.getElementById('result').innerText = 'Please enter a link.';
    }
}

function detectFromPost() {
    const post = document.getElementById('post-input').value;
    if (post) {
        detectText(post);
    } else {
        document.getElementById('result').innerText = 'Please enter a post.';
    }
}

async function detectText(text) {
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const raw = await response.text();
        let data;
        try {
            data = raw ? JSON.parse(raw) : null;
        } catch (e) {
            data = null;
        }

        if (!response.ok) {
            const message = data && data.error ? data.error : `${response.status} ${response.statusText}`;
            document.getElementById('result').innerText = `Server error: ${message}`;
            return;
        }

        if (!data) {
            document.getElementById('result').innerText = 'Server returned invalid JSON.';
            return;
        }

        if (data.error) {
            document.getElementById('result').innerText = data.error;
        } else {
            document.getElementById('result').innerHTML = `
                Result: ${data.result}<br>
                Fake Percentage: ${data.fake_percentage}%<br>
                Real Percentage: ${data.real_percentage}%<br>
                Accuracy: ${data.accuracy}%
            `;
        }
    } catch (error) {
        document.getElementById('result').innerText = 'Error detecting: ' + error.message;
    }
}