// Spinner functionality
document.addEventListener('DOMContentLoaded', function() {
    // Create spinner overlay if it doesn't exist
    if (!document.getElementById('spinner-overlay')) {
        const spinnerOverlay = document.createElement('div');
        spinnerOverlay.id = 'spinner-overlay';
        spinnerOverlay.className = 'fixed inset-0 bg-gray-800 bg-opacity-50 z-50 flex items-center justify-center hidden';
        spinnerOverlay.innerHTML = `
            <div class="bg-white p-5 rounded-lg shadow-lg flex flex-col items-center">
                <div class="spinner-border animate-spin inline-block w-12 h-12 border-4 rounded-full border-blue-500 border-t-transparent" role="status"></div>
                <p class="mt-3 text-gray-700 font-medium">Loading...</p>
            </div>
        `;
        document.body.appendChild(spinnerOverlay);
    }

    // Add event listeners for forms to show spinner on submit
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            showSpinner();
        });
    });

    // Add event listeners for links that should show the spinner
    document.querySelectorAll('a:not([data-no-spinner])').forEach(link => {
        link.addEventListener('click', function() {
            // Don't show spinner for external links or anchors
            if (link.getAttribute('href').startsWith('#') || 
                link.getAttribute('href').startsWith('http') ||
                link.getAttribute('href').startsWith('mailto') ||
                link.getAttribute('href').startsWith('tel') ||
                link.getAttribute('href') === '#' ||
                link.hasAttribute('download')) {
                return;
            }
            showSpinner();
        });
    });

    // Add event listeners for buttons that should show the spinner
    document.querySelectorAll('button:not([type="button"]):not([data-no-spinner])').forEach(button => {
        button.addEventListener('click', function() {
            // Don't show spinner for buttons with type="button"
            if (button.getAttribute('type') !== 'submit' && !button.closest('form')) {
                return;
            }
            showSpinner();
        });
    });
});

// Function to show the spinner
function showSpinner() {
    const spinner = document.getElementById('spinner-overlay');
    if (spinner) {
        spinner.classList.remove('hidden');
    }
}

// Function to hide the spinner
function hideSpinner() {
    const spinner = document.getElementById('spinner-overlay');
    if (spinner) {
        spinner.classList.add('hidden');
    }
}

// Hide spinner when page is loaded
window.addEventListener('load', hideSpinner);

// Show spinner when navigating away from the page
window.addEventListener('beforeunload', function() {
    showSpinner();
});

// Add a global function to control the spinner from anywhere
window.appSpinner = {
    show: showSpinner,
    hide: hideSpinner
};
