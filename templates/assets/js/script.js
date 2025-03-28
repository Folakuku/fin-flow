// Speech Recognition
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('description').value = transcript;
};
recognition.onend = () => {
    document.getElementById('mic-text').textContent = 'Speak';
};

// Form Submission
document.getElementById('submit-btn').addEventListener('click', async (e) => {
    e.preventDefault();
    const description = document.getElementById('description').value;
    if (!description) {
        alert('Please provide a description.');
        return;
    }
    // Show the loading indicator
    document.getElementById('loading').style.display = 'flex';
    // Hide the viewer section
    document.getElementById('viewer-section').style.display = 'none';
    // Simulate a delay (remove this in the real implementation)
    setTimeout(() => {
        // Hide the loading indicator
        document.getElementById('loading').style.display = 'none';
        document.getElementById('viewer-section').style.display = 'block';
        // Clear the existing scene before loading a new model
        while (scene.children.length > 0) {
            scene.remove(scene.children[0]);
        }
        // Load the dummy model
        loadModel('assets/models/box.glb'); // Replace with your dummy model URL
    }, 2000); // 2-second delay
});

// Microphone Button
document.getElementById('mic-btn').addEventListener('click', () => {
    recognition.start();
    document.getElementById('mic-text').textContent = 'Listening...';
});

// 3D Viewer Setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('3d-canvas') });
renderer.setSize(window.innerWidth * 0.8, 500);
renderer.setClearColor(0x000000); // Black background for 3D canvas
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// Position the camera
camera.position.z = 5;

// Placeholder for further JavaScript logic (e.g., form submission, model loading)

function loadModel(url) {
    const loader = new THREE.GLTFLoader();
    loader.load(url, (gltf) => {
        scene.add(gltf.scene);
    });
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Start the animation loop
animate();

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth * 0.8, 500);
});

// Login Form
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        if (data.token) {
            localStorage.setItem('token', data.token);
            alert('Login successful');
            const modal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            modal.hide();
        } else {
            alert('Invalid credentials');
        }
    } catch (error) {
        console.error('Error:', error);
    }
});

// Signup Form
document.getElementById('signup-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('signup-username').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    try {
        const response = await fetch('/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });
        const data = await response.json();
        if (data.token) {
            localStorage.setItem('token', data.token);
            alert('Signup successful');
            const modal = bootstrap.Modal.getInstance(document.getElementById('signupModal'));
            modal.hide();
        } else {
            alert('Signup failed');
        }
    } catch (error) {
        console.error('Error:', error);
    }
});

// ScrollReveal Animations
ScrollReveal().reveal('.hero-section', {
    delay: 200,
    distance: '50px',
    origin: 'bottom',
    easing: 'ease-in-out',
    duration: 800,
    reset: true
});

ScrollReveal().reveal('.video-section', {
    delay: 400,
    distance: '50px',
    origin: 'bottom',
    easing: 'ease-in-out',
    duration: 800,
    reset: true
});

ScrollReveal().reveal('.input-section', {
    delay: 600,
    distance: '50px',
    origin: 'bottom',
    easing: 'ease-in-out',
    duration: 800,
    reset: true
});

ScrollReveal().reveal('#viewer-section', {
    delay: 800,
    distance: '50px',
    origin: 'bottom',
    easing: 'ease-in-out',
    duration: 800,
    reset: true
});
