//---------------------------For the activate dropdown navigation menu---------------------------//
function active() {
    navBar = document.querySelector(".navBar");
    navBar.classList.toggle("active");
}
//---------------------------For the activate target---------------------------//
function activeButton() {
    button = document.querySelector(".image-graph-container");
    button.classList.toggle("activeButton");
}

//---------------------------Form Validation in Login form---------------------------//


let userName, password, flag;


/* Initialize Variables*/
function initializeVars() {
    userName = document.getElementById("username").value;
    password = document.getElementById("password").value;

}
/* Username Validate */
function checkUsername() {
    userName = document.getElementById("username").value;
    let user = "Kastoria";

    if(userName != user){
        flag++;

        //For checkmark
        document.getElementById("validate-mark-username").innerHTML = `<i class="fa fa-exclamation-circle"></i>`;
        document.getElementById("validate-mark-username").style.color="red";
        document.getElementById("validate-mark-username").style.background="transparent";
        //For border bottom.
        document.getElementById("username").style.borderBottom="3px solid red";
    }
    else {
         //For checkmark
         document.getElementById("validate-mark-username").innerHTML = `<i class="fa fa-check-circle"></i>`;
         document.getElementById("validate-mark-username").style.color="green";
         document.getElementById("validate-mark-username").style.background="transparent";
         //For border bottom.
         document.getElementById("username").style.borderBottom="3px solid green";
    }
}
/* Password Validate */
function checkPassword() {
    password = document.getElementById("password").value;

    let pass = 25692907;

    if(password!=pass) {
        flag++;

        //For checkmark
        document.getElementById("validate-mark-password").innerHTML = `<i class="fa fa-exclamation-circle"></i>`;
        document.getElementById("validate-mark-password").style.color="red";
        document.getElementById("validate-mark-password").style.background="transparent";
        //For border bottom.
        document.getElementById("password").style.borderBottom="3px solid red";
    }
    else {
        //For checkmark
        document.getElementById("validate-mark-password").innerHTML = `<i class="fa fa-check-circle"></i>`;
        document.getElementById("validate-mark-password").style.color="green";
        document.getElementById("validate-mark-password").style.background="transparent";
        //For border bottom.
        document.getElementById("password").style.borderBottom="3px solid green";
    }
}

function formValidate() {
    flag = 0;

    checkUsername();
    checkPassword();

    if(flag > 0)
        return false;
}

// PREDICTION ARROWS (FORECASTING.HTML)
let btc_forecast_price, btc_last_price;
let eth_forecast, eth_last;
let bnb_forecast, bnb_last;

btc_forecast_price = document.getElementById("btc_pred").innerHTML;
btc_last_price = document.getElementById("close1").innerHTML;

eth_forecast = document.getElementById("eth_pred").innerHTML;
eth_last = document.getElementById("close2").innerHTML;

bnb_forecast = document.getElementById("bnb_pred").innerHTML;
bnb_last = document.getElementById("close3").innerHTML;


btc_forecast_price = parseFloat(btc_forecast_price);
btc_last_price = parseFloat(btc_last_price);

eth_forecast = parseFloat(eth_forecast);
eth_last = parseFloat(eth_last);

bnb_forecast = parseFloat(bnb_forecast);
bnb_last = parseFloat(bnb_last);




setInterval(upDown, 1000);
setInterval(upDownChange, 2000);

function upDown() {
    //FOR BITCOIN
    if(btc_forecast_price > btc_last_price ) {
        document.getElementById("BTC-up").style.color="white"
        document.getElementById("BTC-up").style.backgroundColor="green"
    }
    else if(btc_forecast_price < btc_last_price ) {
        document.getElementById("BTC-down").style.color="white";
        document.getElementById("BTC-down").style.backgroundColor="red";
    }
    else {
        document.getElementById("BTC-up").style.color="white";
        document.getElementById("BTC-down").style.color="white";
    }
    
    //FOR ETHEREUM
    if(eth_forecast > eth_last ) {
        document.getElementById("ETH-up").style.color="white"
        document.getElementById("ETH-up").style.backgroundColor="green"
    }
    else if(eth_forecast < eth_last ) {
        document.getElementById("ETH-down").style.color="white";
        document.getElementById("ETH-down").style.backgroundColor="red";
    }
    else {
        document.getElementById("ETH-up").style.color="white";
        document.getElementById("ETH-down").style.color="white";
    }
    
    //FOR BINANCE
    if(bnb_forecast > bnb_last ) {
        document.getElementById("BNB-up").style.color="white"
        document.getElementById("BNB-up").style.backgroundColor="green"
    }
    else if(bnb_forecast < bnb_last ) {
        document.getElementById("BNB-down").style.color="white";
        document.getElementById("BNB-down").style.backgroundColor="red";
    }
    else {
        document.getElementById("BNB-up").style.color="white";
        document.getElementById("BNB-down").style.color="white";
    }
}

function upDownChange() {
    //FOR BITCOIN
    if(btc_forecast_price > btc_last_price ) {
        document.getElementById("BTC-up").style.backgroundColor="transparent";
    }
    else if(btc_forecast_price < btc_last_price ) {
        document.getElementById("BTC-down").style.backgroundColor="transparent";
    }
    else {
        document.getElementById("BTC-down").style.backgroundColor="transparent";
    }
    
    //FOR ETHEREUM
    if(eth_forecast > eth_last ) {
        document.getElementById("ETH-up").style.backgroundColor="transparent";
    }
    else if(eth_forecast < eth_last ) {
        document.getElementById("ETH-down").style.backgroundColor="transparent";
    }
    else {
        document.getElementById("ETH-down").style.backgroundColor="transparent";
    }
    
    //FOR BINANCE
    if(bnb_forecast > bnb_last ) {
        document.getElementById("BNB-up").style.backgroundColor="transparent";
    }
    else if(bnb_forecast < bnb_last ) {
        document.getElementById("BNB-down").style.backgroundColor="transparent";
    }
    else {
        document.getElementById("BNB-down").style.backgroundColor="transparent";
    }   
}
