const rain = document.querySelector(".rain");

if (rain) {

    for (let i = 0; i < 100; i++) {

        const drop = document.createElement("span");

        drop.style.left = Math.random() * 100 + "%";

        drop.style.animationDuration = (0.5 + Math.random() * 0.8) + "s";

        drop.style.animationDelay = Math.random() * 2 + "s";

        drop.style.height = (70 + Math.random() * 60) + "px";

        drop.style.opacity = 0.3 + Math.random() * 0.7;

        rain.appendChild(drop);
    }

}