// Glavni parametri
suction_cup_dia = 80;
suction_cup_height = 15;
arm_length = 60;
arm_dia = 10;
phone_holder_width = 70;
phone_holder_height = 120;
phone_holder_depth = 15;
grip_width = 8;

t = $t; // Za animacijo od 0..1

module suction_cup() {
    color("SlateGray"){

    // Zgornji rob
    translate([0, 0, suction_cup_height-2])
        cylinder(d=suction_cup_dia-10, h=4, $fn=60);
    
    // Spodnji nivo spodnjega sklepa
    translate([0, 0, suction_cup_height])
        cylinder(d=20, h=8, $fn=30);
    
    // Animiran vakumski del nosilca
    difference(){
    // Glavno držalo
    cylinder(d=suction_cup_dia, h=suction_cup_height, $fn=80);
    
    // Notranji cilinder
    translate([0, 0, -5 + 4*t])
        cylinder(d=suction_cup_dia-10, h=4, $fn=60);
}   
    // Ročka za nastavitev vakuma
    translate([-8, 36, 5])
        rotate([-45 - 45*t, 0, 0])
            cube([15, 3, 25]);
 }
}
 
module articulating_arm() {
    // Spodnji sklep
    color("Silver")
        translate([0, 0, suction_cup_height+8])
            sphere(d=18, $fn=50);
    
    // Roka
    color("SlateGray")
    translate([0, 0, suction_cup_height+8])
        rotate([45, 0, 0])
            cylinder(d=arm_dia, h=arm_length, $fn=20);
    
    // Zgornji sklep
    color("Silver")
        translate([0, -arm_length*sin(45), suction_cup_height+8+arm_length*cos(45)])
            sphere(d=18, $fn=50);
}

module phone_holder() {
    arm_end_x = 0;
    arm_end_y = -arm_length*sin(45);
    arm_end_z = suction_cup_height+8+arm_length*cos(45);
    
    // Držalo telefona postavljeno na konec ročke
    translate([-arm_length, arm_end_y - 10, arm_end_z]) {
        rotate([0, 90, 0]) { 
            // Okvir telefona
            difference(){
                color("gray")
                translate([-phone_holder_width/2, -phone_holder_depth/2+5 , 0])
                cube([phone_holder_width, phone_holder_depth - 7, phone_holder_height]);
            
            // Zaslon telefona
                color("lightblue")
                translate([-phone_holder_width/2+grip_width +2, -phone_holder_depth/2+3, grip_width])
                cube([phone_holder_width-2*grip_width - 2, phone_holder_depth-12, 
                phone_holder_height-2*grip_width]);
            }
            
            
            // Zgornja klešča
            color("SlateGray")
                translate([-phone_holder_width/2-5, -phone_holder_depth/2, phone_holder_height/2-15])
                    cube([grip_width+5, phone_holder_depth + 3, 30]);
            
            // Spodnja klešča
            color("SlateGray")
                translate([phone_holder_width/2 - 5, -phone_holder_depth/2, phone_holder_height/2-15])
                    cube([grip_width+5, phone_holder_depth + 3, 30]);
            
            // Povezava med kleščama
            color("SlateGray")
            translate([-30, 5.5, 47])
            cube([60, 5, 26]);
            
            // Kamera
            union(){
                color("DimGray")
                translate([-30, 5.5, 5])
                cube([25, 2, 25]);
            
                color("Black")
                translate([-31, 5, 4])
                cube([27, 2, 27]);
            }
            
            color("Black"){
                translate([-24, 8, 11])
                rotate([90,0,0])
                cylinder(d=10, h=0.5, $fn=60);
            
                translate([-11, 8, 11])
                rotate([90,0,0])
                cylinder(d=10, h=0.5, $fn=60);
            }
            
            color("LightCyan"){
                translate([-11, 8.1, 11])
                rotate([90,0,0])
                cylinder(d=8, h=0.1, $fn=60);
            
                translate([-24, 8.1, 11])
                rotate([90,0,0])
                cylinder(d=8, h=0.1, $fn=60);
            }
        }
    }
}

// Sestavljanje z uporabo funkcije Union
difference() {
    union() {
        suction_cup();
        articulating_arm();
        phone_holder();
    }
    
    // Prostor zrak
    translate([0, 0, -2])
        cylinder(d=6, h=suction_cup_height+12, $fn=20);
}