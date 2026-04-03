//VERSION=3

function setup(){
    return {
        input: ["B04", "B03", "B02", "dataMask"],
        output: { bands : 4 },
    };
}

function evaluatePixel(smp){
    let f = 1.75;

    return [
            f*Math.cbrt(0.6 * smp.B04 - 0.035),
            f*Math.cbrt(0.6 * smp.B03 - 0.035),
            f*Math.cbrt(0.6 * smp.B02 - 0.035),
            smp.dataMask,
        ];
}