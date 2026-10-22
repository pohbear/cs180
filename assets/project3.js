// Keep the outer button fixed so rotating/shrinking its image never loses hover.
// Native buttons also supply Enter/Space activation and touch support.
document.querySelectorAll('.illusion-toggle').forEach(button => {
  button.addEventListener('click', () => {
    const active = button.classList.toggle('is-active');
    button.setAttribute('aria-pressed', String(active));
  });
});

// Print code snippets in full; restore the visitor's choices afterward.
let openBeforePrint = [];
window.addEventListener('beforeprint', () => {
  openBeforePrint = [...document.querySelectorAll('details')].map(details => [details, details.open]);
  openBeforePrint.forEach(([details]) => { details.open = true; });
});
window.addEventListener('afterprint', () => {
  openBeforePrint.forEach(([details, wasOpen]) => { details.open = wasOpen; });
});
