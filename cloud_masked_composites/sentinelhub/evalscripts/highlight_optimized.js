function setup(){
    return {
        input: ["B04", "B03", "B02"],
        output: { bands : 3 }
    };
}

function evaluatePixel(smp){
    return [Math.cbrt(0.6 * smp.B04 - 0.035), Math.cbrt(0.6 * smp.B03 - 0.035), Math.cbrt(0.6 * smp.B02 - 0.035)];
}