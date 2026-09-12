#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "PhotonSensitiveDetector.hh"

#include "G4SDManager.hh"

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4NistManager.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include "G4SystemOfUnits.hh"



// Constructor


DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fAerogelLogical(nullptr),
      fDetectorLogical(nullptr),
      fRefractiveIndex(1.026)
{
    
    // Detector geometry parameters
   

    // Aerogel:
    // 100 mm x 100 mm x 20 mm
    fAerogelThickness = 20.0 * mm;
    fAerogelSize      = 100.0 * mm;

    // Detector:
    // 200 mm x 200 mm
    fDetectorSize     = 200.0 * mm;

    // Distance from downstream aerogel face to front face of detector
    
    fDetectorDistance = 200.0 * mm;
}



// Destructor


DetectorConstruction::~DetectorConstruction()
{
}



// Construct detector


G4VPhysicalVolume*
DetectorConstruction::Construct()
{
    G4NistManager* nist =
        G4NistManager::Instance();


    // 1. WORLD
    
    G4double worldSize = 3.0 * m;

    G4Material* worldMaterial =
        nist->FindOrBuildMaterial("G4_AIR");

    G4Box* worldSolid =
        new G4Box(
            "World",
            worldSize / 2.0,
            worldSize / 2.0,
            worldSize / 2.0
        );

    G4LogicalVolume* worldLogical =
        new G4LogicalVolume(
            worldSolid,
            worldMaterial,
            "World"
        );

    G4VPhysicalVolume* worldPhysical =
        new G4PVPlacement(
            nullptr,
            G4ThreeVector(0, 0, 0),
            worldLogical,
            "World",
            nullptr,
            false,
            0,
            true
        );


    
    // 2. WORLD OPTICAL PROPERTIES
    
    // Optical photons need an index of refraction for the medium through which they propagate.
    // We use approximately:  n_air = 1.0 over the optical wavelength range 300-650 nm.
    

    const G4int nAirEntries = 2;

    G4double airPhotonEnergy[nAirEntries] =
    {
        1.91 * eV,   // ~650 nm
        4.13 * eV    // ~300 nm
    };

    G4double airRefractiveIndex[nAirEntries] =
    {
        1.0,
        1.0
    };

    G4MaterialPropertiesTable* airMPT =
        new G4MaterialPropertiesTable();

    airMPT->AddProperty(
        "RINDEX",
        airPhotonEnergy,
        airRefractiveIndex,
        nAirEntries
    );

    worldMaterial->SetMaterialPropertiesTable(airMPT);


    // 3. AEROGEL MATERIAL
    
    // Simplified aerogel density.
    
    G4double aerogelDensity =
        0.2 * g / cm3;



    G4Material* aerogel =
        new G4Material(
            "Aerogel",
            aerogelDensity,
            2
        );


    G4Element* Si =
        nist->FindOrBuildElement("Si");

    G4Element* O =
        nist->FindOrBuildElement("O");


    aerogel->AddElement(
        Si,
        1
    );

    aerogel->AddElement(
        O,
        2
    );


    // 4. AEROGEL OPTICAL PROPERTIES
   
    // Optical wavelength range:  300 nm --> 650 nm
    
    //     650 nm --> ~1.91 eV
    //     300 nm --> ~4.13 eV
    

    const G4int nAerogelEntries = 2;

    G4double aerogelPhotonEnergy[nAerogelEntries] =
    {
        1.91 * eV,   // ~650 nm
        4.13 * eV    // ~300 nm
    };

   G4double aerogelRefractiveIndex[nAerogelEntries] =
    {
        fRefractiveIndex,
        fRefractiveIndex
    };

    G4MaterialPropertiesTable* aerogelMPT =
        new G4MaterialPropertiesTable();

    aerogelMPT->AddProperty(
        "RINDEX",
        aerogelPhotonEnergy,
        aerogelRefractiveIndex,
        nAerogelEntries
    );
        
    // Rayleigh scattering length
    //
    //     Λ(λ) = λ^4 / C
    

    G4double aerogelRayleigh[nAerogelEntries] =
    {
        35.7 * cm,   // ~650 nm (1.91 eV)
        1.62 * cm    // ~300 nm (4.13 eV)
    };

    aerogelMPT->AddProperty(
        "RAYLEIGH",
        aerogelPhotonEnergy,
        aerogelRayleigh,
        nAerogelEntries
    );


    
    // Absorption length (placeholder)
    

    G4double aerogelAbsLength[nAerogelEntries] =
    {
        100.0 * cm,
        100.0 * cm
    };

    aerogelMPT->AddProperty(
        "ABSLENGTH",
        aerogelPhotonEnergy,
        aerogelAbsLength,
        nAerogelEntries
    );

    aerogel->SetMaterialPropertiesTable(
        aerogelMPT
    );


    
    // 5. AEROGEL GEOMETRY
    
    // Dimensions:
    //
    //     X = 100 mm
    //     Y = 100 mm
    //     Z = 20 mm
    //
    // The aerogel is centered at:
    //
    //     (0, 0, 0)


    G4Box* aerogelSolid =
        new G4Box(
            "Aerogel",
            fAerogelSize / 2.0,
            fAerogelSize / 2.0,
            fAerogelThickness / 2.0
        );


    fAerogelLogical =
        new G4LogicalVolume(
            aerogelSolid,
            aerogel,
            "Aerogel"
        );


    new G4PVPlacement(
        nullptr,
        G4ThreeVector(0, 0, 0),
        fAerogelLogical,
        "Aerogel",
        worldLogical,
        false,
        0,
        true
    );


  

G4Material* detectorMaterial =
    new G4Material(
        "PhotonDetector",
        1.0 * g / cm3,
        1
    );

G4Element* detectorElement =
    nist->FindOrBuildElement("H");

detectorMaterial->AddElement(
    detectorElement,
    1
);

const G4int nDetectorEntries = 2;

G4double detectorPhotonEnergy[nDetectorEntries] =
{
    1.91 * eV,
    4.13 * eV
};

G4double detectorRefractiveIndex[nDetectorEntries] =
{
    1.0,
    1.0
};

G4MaterialPropertiesTable* detectorMPT =
    new G4MaterialPropertiesTable();

detectorMPT->AddProperty(
    "RINDEX",
    detectorPhotonEnergy,
    detectorRefractiveIndex,
    nDetectorEntries
);

detectorMaterial->SetMaterialPropertiesTable(
    detectorMPT
);


   

    G4double detectorThickness =
        1.0 * mm;


    G4Box* detectorSolid =
        new G4Box(
            "Detector",
            fDetectorSize / 2.0,
            fDetectorSize / 2.0,
            detectorThickness / 2.0
        );


    fDetectorLogical =
        new G4LogicalVolume(
            detectorSolid,
            detectorMaterial,
            "Detector"
        );


   

    G4double aerogelDownstreamZ =
        fAerogelThickness / 2.0;


    G4double detectorFrontZ =
        aerogelDownstreamZ
        + fDetectorDistance;


    G4double detectorCenterZ =
        detectorFrontZ
        + detectorThickness / 2.0;


    new G4PVPlacement(
        nullptr,
        G4ThreeVector(
            0,
            0,
            detectorCenterZ
        ),
        fDetectorLogical,
        "Detector",
        worldLogical,
        false,
        0,
        true
    );


   
    // 9. VISUALIZATION
   
    G4VisAttributes* aerogelVis =
        new G4VisAttributes(
            G4Colour(
                0.3,
                0.7,
                1.0
            )
        );

    aerogelVis->SetForceSolid(true);

    fAerogelLogical->SetVisAttributes(
        aerogelVis
    );


 
    // Detector visualization


    G4VisAttributes* detectorVis =
        new G4VisAttributes(
            G4Colour(
                0.7,
                0.7,
                0.7
            )
        );

    detectorVis->SetForceSolid(true);

    fDetectorLogical->SetVisAttributes(
        detectorVis
    );



    // World visualization

    worldLogical->SetVisAttributes(
        G4VisAttributes::GetInvisible()
    );


   
    // 10. RETURN WORLD
 

    return worldPhysical;
}


// Construct sensitive detector

void DetectorConstruction::ConstructSDandField()
{
    G4SDManager* sdManager =
        G4SDManager::GetSDMpointer();

    PhotonSensitiveDetector* photonSD =
        new PhotonSensitiveDetector(
            "PhotonSensitiveDetector"
        );

    sdManager->AddNewDetector(
        photonSD
    );

    fDetectorLogical->SetSensitiveDetector(
        photonSD
    );
}

void DetectorConstruction::SetRefractiveIndex(G4double n)
{
    fRefractiveIndex = n;
}