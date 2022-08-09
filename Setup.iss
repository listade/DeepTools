#define AppName "DeepTools"
#define AppVersion "1.0"
#define AppPublisher "Sholukh Egor"

#define python "python-3.7.9-amd64.exe"
#define cuda "cuda_11.6.2_511.65_windows.exe"
#define cudnn "cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive"
#define cudnn_zip cudnn + ".zip"

[Setup]
AppId={{7b281e88-150e-46f6-a11d-c46d70783e4e}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=install
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir="."
SetupLogging=yes
ChangesEnvironment=yes
RestartIfNeededByRun=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "app\*"; DestDir: "{app}\app"; Flags: recursesubdirs ignoreversion
Source: "cfg\*"; DestDir: "{app}\cfg"; Flags: recursesubdirs ignoreversion
Source: "data\*"; DestDir: "{app}\data"; Flags: recursesubdirs ignoreversion

Source: "{src}\weights\*"; DestDir: "{app}\weights"; Flags: external
Source: "{src}\input\*"; DestDir: "{app}\input"; Flags: external

[Run]
Filename: "{src}\dist\{#python}"; Parameters: "/passive PrependPath=1"; Flags: waituntilterminated; StatusMsg: "Installing.. {#python}"
Filename: "{src}\dist\{#cuda}"; Parameters: "-s"; Flags: waituntilterminated; StatusMsg: "Installing.. {#cuda}"
Filename: "powershell"; Parameters: "-command ""Expand-Archive -Force -Verbose '{src}\dist\{#cudnn_zip}' '{app}\' "" "; Flags: waituntilterminated; StatusMsg: "Extracting.. {#cudnn_zip}"
Filename: "{src}\build.bat"; Parameters: """{app}\env"""; Flags: waituntilterminated; StatusMsg: "Installing.. packages"
Filename: "{cmd}"; Parameters: "/k {src}\test.bat"; WorkingDir: "{app}"; Description: "Run test"; Flags: postinstall runascurrentuser

[UninstallRun]
Filename: "{src}\dist\{#Python}"; RunOnceId: "RemovePython";  Parameters: "/uninstall"; Flags: waituntilterminated; StatusMsg: "Uninstalling.. {#python}"


[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\{#cudnn}"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}\{#cudnn}\bin"; Check: NeedsAddPath(ExpandConstant('{app}\{#cudnn}\bin'))

[Code]

procedure CurStepChanged(CurStep: TSetupStep);
var
  logfilepathname, logfilename, newfilepathname: string;
begin
  logfilepathname := ExpandConstant('{log}');
  logfilename := ExtractFileName(logfilepathname);
  newfilepathname := ExpandConstant('{src}\') + logfilename;

  if CurStep = ssDone then
  begin
    FileCopy(logfilepathname, newfilepathname, false);
  end;
end;

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  { look for the path with leading and trailing semicolon }
  { Pos() returns 0 if not found }
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;